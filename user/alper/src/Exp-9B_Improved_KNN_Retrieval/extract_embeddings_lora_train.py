"""Extract 768-d ArcFace embeddings from the LoRA-tuned DINOv2 backbone
over the ~1.4M PlantCLEF 2024 training images.

Output schema matches project_embeddings.py --mode train, so
build_faiss_index_512.py consumes it unchanged.

  Output parquet columns: [image_name, species_id (int32), emb768 (float32 768d)]

Sharding: ~70K images per parquet (-> ~20 shards for 1.4M). Shards are written
to the same sample_id=* partition layout as the original cached embeddings,
for compatibility with build_faiss_index_512.py's glob.

Usage:
  python extract_embeddings_lora_train.py \
    --data-dir /storage/.../train/PlantCLEF2024 \
    --backbone-path /storage/.../vit_base_patch14_reg4_dinov2.pth \
    --checkpoint /storage/.../exp9b_knn/arcface_lora/lora_head_final.pt \
    --species-map /path/to/class_to_species_id.json \
    --out-dir /storage/.../exp9b_knn/embeddings_arcface_lora/train
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import timm
import torch
from PIL import ImageFile
from torch.utils.data import DataLoader
from torchvision import transforms

from lora import inject_lora_into_attention, load_lora_state

ImageFile.LOAD_TRUNCATED_IMAGES = True
from train_arcface_lora import (
    ArcFaceHead,
    ImageFolderWithSpeciesId,
    JPEGCompression,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_NAME,
    NUM_CLASSES,
    IN_DIM,
    load_species_to_class,
)


def build_eval_transform(image_size: int, jpeg_quality: int) -> transforms.Compose:
    tx = [
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(image_size),
    ]
    if jpeg_quality > 0:
        tx.append(JPEGCompression(quality=jpeg_quality, subsampling=1))
    tx += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(tx)


def load_model(backbone_path: str, checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {})
    print(f"[model] checkpoint epoch={ckpt.get('epoch')} step={ckpt.get('step')}")
    print(f"[model] cfg: lora_r={cfg.get('lora_r')} alpha={cfg.get('lora_alpha')} "
          f"out_dim={cfg.get('out_dim')} image_size={cfg.get('image_size')}")

    backbone = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=NUM_CLASSES,
        checkpoint_path=backbone_path,
    ).to(device)
    inject_lora_into_attention(
        backbone,
        r=int(cfg.get("lora_r", 8)),
        alpha=float(cfg.get("lora_alpha", 16.0)),
        dropout=0.0,
        include_proj=bool(cfg.get("lora_on_proj", False)),
    )
    backbone = backbone.to(device)
    load_lora_state(backbone, ckpt["lora"])
    backbone.eval()

    head = ArcFaceHead(in_dim=IN_DIM, out_dim=int(cfg.get("out_dim", 768))).to(device)
    head.load_state_dict(ckpt["head"])
    head.eval()

    return backbone, head, cfg


@torch.no_grad()
def extract(
    backbone,
    head,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    all_emb = []
    all_lbl = []
    all_name = []

    for i, (imgs, labels, paths) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            feats = backbone.forward_features(imgs)
            cls = feats[:, 0, :]
            z = head(cls)
        all_emb.append(z.float().cpu().numpy())
        all_lbl.append(labels.numpy())
        all_name.extend(os.path.basename(p) for p in paths)
        if (i + 1) % 20 == 0:
            n_done = sum(e.shape[0] for e in all_emb)
            print(f"[extract] {i + 1} batches, {n_done} images")
    emb = np.concatenate(all_emb, axis=0)
    lbl = np.concatenate(all_lbl, axis=0).astype(np.int32)
    return emb, lbl, all_name


class ImageFolderWithPaths(ImageFolderWithSpeciesId):
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Project PlantCLEF train images through LoRA+ArcFace head"
    )
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--backbone-path", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--species-map", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--shard-size", type=int, default=70000)
    ap.add_argument("--image-size", type=int, default=None,
                    help="Override image size from checkpoint config.")
    ap.add_argument("--jpeg-quality", type=int, default=None,
                    help="Override JPEG quality from checkpoint config.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}")

    species_to_class = load_species_to_class(args.species_map)
    print(f"[init] {len(species_to_class)} species in mapping")

    backbone, head, cfg = load_model(args.backbone_path, args.checkpoint, device)

    img_size = args.image_size if args.image_size is not None else int(cfg.get("image_size", 518))
    jpeg_q = args.jpeg_quality if args.jpeg_quality is not None else int(cfg.get("jpeg_quality", 85))
    print(f"[init] image_size={img_size} jpeg_q={jpeg_q}")

    transform = build_eval_transform(img_size, jpeg_q)
    dataset = ImageFolderWithPaths(
        root=args.data_dir,
        transform=transform,
        species_to_class=species_to_class,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    print(f"[init] n_samples={len(dataset)} batches={len(loader)}")

    t0 = time.time()
    emb, lbl, names = extract(backbone, head, loader, device)
    elapsed = time.time() - t0
    print(f"[extract] done {len(emb)} embeddings in {elapsed:.1f}s "
          f"({len(emb)/elapsed:.1f} img/s)  dim={emb.shape[1]}")

    inv_species_map = {cls: sid for sid, cls in species_to_class.items()}

    os.makedirs(args.out_dir, exist_ok=True)
    n = len(emb)
    shard_size = args.shard_size
    n_shards = (n + shard_size - 1) // shard_size
    print(f"[out] writing {n_shards} shards of up to {shard_size} rows each")

    for shard in range(n_shards):
        s = shard * shard_size
        e = min(n, s + shard_size)
        shard_dir = os.path.join(args.out_dir, f"sample_id={shard}")
        os.makedirs(shard_dir, exist_ok=True)
        df = pd.DataFrame(
            {
                "image_name": names[s:e],
                "species_id": [inv_species_map[int(lbl[i])] for i in range(s, e)],
                "emb768": list(emb[s:e]),
            }
        )
        df["species_id"] = df["species_id"].astype("int32")
        out_path = os.path.join(shard_dir, f"part-{shard:05d}.snappy.parquet")
        df.to_parquet(out_path, index=False, compression="snappy")
        print(f"[out]   wrote {out_path}  rows={len(df)}")

    print(f"[done] train embedding extraction complete under {args.out_dir}")


if __name__ == "__main__":
    main()

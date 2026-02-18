#!/usr/bin/env python

import os
import glob
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset, DataLoader

from Cross_Region_Transformer_v3 import CrossRegionTransformer


def freeze_by_name(model: nn.Module, trainable_substrings):
    trainable_substrings = [s.strip() for s in trainable_substrings if s.strip()]
    for name, p in model.named_parameters():
        p.requires_grad = False
        for s in trainable_substrings:
            if s in name:
                p.requires_grad = True
                break


def load_pseudo_map(pseudo_parquet: str):
    p = pq.read_table(
        pseudo_parquet,
        columns=["image_name", "k", "pos_idx_i32", "pos_prob_f16"],
    ).to_pandas()

    pseudo = {}
    for _, r in p.iterrows():
        k = int(r["k"])
        idx = np.frombuffer(r["pos_idx_i32"], dtype=np.int32, count=k).astype(np.int64)
        prob = np.frombuffer(r["pos_prob_f16"], dtype=np.float16, count=k).astype(np.float32)
        pseudo[r["image_name"]] = (idx, prob)

    print("pseudo rows:", len(p))
    return pseudo


class Step3BSelfTrainIterable(IterableDataset):
    """
    Streams merged inputs from sharded root and joins with pseudo labels via in memory dict.

    Yields:
      tiles: [T, D] float32
      meta:  [4] float32
      pos_idx: 1D int64
      pos_prob: 1D float32
      teacher_blob: optional, bytes, if teacher is requested
    """

    def __init__(
        self,
        merged_root: str,
        pseudo_map: dict,
        pattern: str = "shard_*/data.parquet",
        num_tiles: int = 16,
        emb_dim: int = 768,
        want_teacher: bool = False,
        batch_rows: int = 512,
    ):
        super().__init__()
        self.merged_root = merged_root
        self.pseudo_map = pseudo_map
        self.pattern = pattern
        self.num_tiles = num_tiles
        self.emb_dim = emb_dim
        self.want_teacher = want_teacher
        self.batch_rows = batch_rows

        self.files = sorted(glob.glob(os.path.join(merged_root, pattern)))
        if not self.files:
            raise ValueError(f"No shard parquet files found under {merged_root} with pattern {pattern}")

    def __iter__(self):
        cols = ["image_name", "tile_embeddings_f16", "metadata_tensor_f32"]
        if self.want_teacher:
            cols.append("teacher_image_probs_f16")

        for fp in self.files:
            d = ds.dataset(fp, format="parquet")
            scanner = d.scanner(columns=cols, batch_size=self.batch_rows)
            for rb in scanner.to_reader():
                names = rb.column("image_name").to_pylist()
                tile_blobs = rb.column("tile_embeddings_f16").to_pylist()
                meta_lists = rb.column("metadata_tensor_f32").to_pylist()
                if self.want_teacher:
                    teacher_blobs = rb.column("teacher_image_probs_f16").to_pylist()
                else:
                    teacher_blobs = None

                for i, img in enumerate(names):
                    if img not in self.pseudo_map:
                        continue

                    idx, prob = self.pseudo_map[img]

                    tiles = np.frombuffer(tile_blobs[i], dtype=np.float16).astype(np.float32, copy=False)
                    tiles = tiles.reshape(self.num_tiles, self.emb_dim)

                    meta = np.asarray(meta_lists[i], dtype=np.float32).copy()

                    if self.want_teacher:
                        tblob = teacher_blobs[i]
                        yield (
                            torch.from_numpy(tiles),
                            torch.from_numpy(meta),
                            torch.from_numpy(idx),
                            torch.from_numpy(prob),
                            tblob,
                        )
                    else:
                        yield (
                            torch.from_numpy(tiles),
                            torch.from_numpy(meta),
                            torch.from_numpy(idx),
                            torch.from_numpy(prob),
                        )


def collate_varlen(batch):
    if len(batch[0]) == 5:
        tiles = torch.stack([b[0] for b in batch], dim=0)
        meta = torch.stack([b[1] for b in batch], dim=0)
        idx_list = [b[2] for b in batch]
        prob_list = [b[3] for b in batch]
        teacher_blobs = [b[4] for b in batch]
        return tiles, meta, idx_list, prob_list, teacher_blobs

    tiles = torch.stack([b[0] for b in batch], dim=0)
    meta = torch.stack([b[1] for b in batch], dim=0)
    idx_list = [b[2] for b in batch]
    prob_list = [b[3] for b in batch]
    return tiles, meta, idx_list, prob_list


def masked_selftrain_loss(logits: torch.Tensor, idx_list, prob_list, neg_sample_k: int = 256):
    B, C = logits.shape
    losses = []

    for i, (idx, prob) in enumerate(zip(idx_list, prob_list)):
        if idx.numel() == 0:
            continue

        idx = idx.to(logits.device, non_blocking=True)
        prob = prob.to(logits.device, non_blocking=True)

        pos_logits = logits[i, idx]
        pos_loss = F.binary_cross_entropy_with_logits(pos_logits, prob, reduction="mean")

        k = min(int(neg_sample_k), C - int(idx.numel()))
        if k > 0:
            neg_idx = torch.randint(0, C, (k,), device=logits.device)

            if idx.numel() < C:
                pos_mask = torch.zeros(C, dtype=torch.bool, device=logits.device)
                pos_mask[idx] = True
                neg_idx = neg_idx[~pos_mask[neg_idx]]
                if neg_idx.numel() < k:
                    extra = torch.randint(0, C, (k - neg_idx.numel(),), device=logits.device)
                    neg_idx = torch.cat([neg_idx, extra], dim=0)

            neg_idx = neg_idx[:k]
            neg_logits = logits[i, neg_idx]
            neg_targets = torch.zeros_like(neg_logits)
            neg_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_targets, reduction="mean")
        else:
            neg_loss = pos_loss * 0.0

        losses.append(pos_loss + neg_loss)

    if len(losses) == 0:
        return logits.sum() * 0.0

    return torch.stack(losses).mean()


def masked_teacher_prior_loss(
    logits: torch.Tensor,
    teacher_blobs,
    num_classes: int,
    teacher_conf_thr: float,
    neg_sample_k: int,
):
    B, C = logits.shape
    if C != num_classes:
        raise ValueError("num_classes mismatch")

    device = logits.device
    teacher = torch.empty((B, C), dtype=torch.float32, device=device)

    for i, blob in enumerate(teacher_blobs):
        t = np.frombuffer(blob, dtype=np.float16).astype(np.float32, copy=False)
        if t.shape[0] != C:
            raise ValueError("teacher length mismatch")
        teacher[i] = torch.from_numpy(t).to(device)

    pos_mask = teacher > float(teacher_conf_thr)
    if pos_mask.any():
        pos_logits = logits[pos_mask]
        pos_targets = teacher[pos_mask]
        pos_loss = F.binary_cross_entropy_with_logits(pos_logits, pos_targets, reduction="mean")
    else:
        pos_loss = logits.sum() * 0.0

    if neg_sample_k > 0:
        neg_mask = ~pos_mask
        neg_losses = []
        for i in range(B):
            zero_idx = torch.nonzero(neg_mask[i], as_tuple=False).squeeze(1)
            if zero_idx.numel() == 0:
                continue
            k = min(int(neg_sample_k), int(zero_idx.numel()))
            sel = zero_idx[torch.randint(0, int(zero_idx.numel()), (k,), device=device)]
            neg_logits = logits[i, sel]
            neg_targets = torch.zeros_like(neg_logits)
            neg_losses.append(F.binary_cross_entropy_with_logits(neg_logits, neg_targets, reduction="mean"))
        neg_loss = torch.stack(neg_losses).mean() if len(neg_losses) > 0 else (logits.sum() * 0.0)
    else:
        neg_loss = logits.sum() * 0.0

    return pos_loss + neg_loss


def co_graph_loss_from_probs(probs, ei, ej, ew, max_edges: int):
    if ei is None or ej is None or ew is None:
        return probs.sum() * 0.0
    if ei.numel() == 0:
        return probs.sum() * 0.0

    E = ei.numel()
    if max_edges > 0 and E > max_edges:
        sel = torch.randint(0, E, (max_edges,), device=probs.device)
        ei_s = ei[sel]
        ej_s = ej[sel]
        ew_s = ew[sel]
    else:
        ei_s, ej_s, ew_s = ei, ej, ew

    pi = probs.index_select(dim=1, index=ei_s)
    pj = probs.index_select(dim=1, index=ej_s)
    diff2 = (pi - pj) ** 2
    w = ew_s.view(1, -1)
    return (diff2 * w).mean()


def save_latest_checkpoint(path, epoch, run_stamp, model, optimizer, epoch_loss, best_loss, args):
    payload = {
        "epoch": int(epoch),
        "run_stamp": run_stamp,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch_loss": float(epoch_loss),
        "best_epoch_loss": float(best_loss),
        "args": vars(args),
    }
    torch.save(payload, path)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--merged-root", required=True)
    ap.add_argument("--pattern", type=str, default="shard_*/data.parquet")
    ap.add_argument("--pseudo-parquet", required=True)

    ap.add_argument("--init-ckpt", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--batch-rows", type=int, default=512)

    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)

    ap.add_argument("--freeze-epochs", type=int, default=2)
    ap.add_argument("--trainable-names", type=str, default="classifier,fc,head,output,metadata")

    ap.add_argument("--neg-sample-k", type=int, default=256)

    ap.add_argument("--lambda-teacher", type=float, default=0.0)
    ap.add_argument("--teacher-conf-thr", type=float, default=0.02)
    ap.add_argument("--teacher-neg-sample-k", type=int, default=128)

    ap.add_argument("--co-graph-npz", type=str, default="")
    ap.add_argument("--lambda-co", type=float, default=0.0)
    ap.add_argument("--co-edges-per-batch", type=int, default=200000)

    ap.add_argument("--num-tiles", type=int, default=16)
    ap.add_argument("--emb-dim", type=int, default=768)
    ap.add_argument("--num-classes", type=int, default=7806)

    ap.add_argument("--resume", type=str, default="")

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    os.makedirs(args.out_dir, exist_ok=True)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M")
    print("run stamp:", run_stamp)

    pseudo_map = load_pseudo_map(args.pseudo_parquet)

    want_teacher = args.lambda_teacher > 0.0
    ds_iter = Step3BSelfTrainIterable(
        merged_root=args.merged_root,
        pseudo_map=pseudo_map,
        pattern=args.pattern,
        num_tiles=args.num_tiles,
        emb_dim=args.emb_dim,
        want_teacher=want_teacher,
        batch_rows=args.batch_rows,
    )

    loader = DataLoader(
        ds_iter,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=collate_varlen,
    )

    model = CrossRegionTransformer(
        input_dim=args.emb_dim,
        hidden_dim=512,
        num_heads=8,
        num_layers=4,
        num_classes=args.num_classes,
        metadata_dim=4,
        dropout=0.1,
        use_attention_pooling=True,
    ).to(device)

    sd = torch.load(args.init_ckpt, map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd, strict=True)

    co_ei = co_ej = co_ew = None
    if args.co_graph_npz:
        g = np.load(args.co_graph_npz)
        co_ei = torch.from_numpy(g["neigh_i"].astype(np.int64)).to(device)
        co_ej = torch.from_numpy(g["neigh_j"].astype(np.int64)).to(device)
        co_ew = torch.from_numpy(g["neigh_w"].astype(np.float32)).to(device)
        print("loaded co graph edges:", int(co_ei.numel()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    start_epoch = 1
    best_loss = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        if "model_state_dict" not in ckpt or "optimizer_state_dict" not in ckpt:
            raise ValueError("Resume checkpoint missing model_state_dict or optimizer_state_dict")

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1

        if "best_epoch_loss" in ckpt:
            best_loss = float(ckpt["best_epoch_loss"])
        elif "epoch_loss" in ckpt:
            best_loss = float(ckpt["epoch_loss"])

        if "run_stamp" in ckpt:
            run_stamp = str(ckpt["run_stamp"])

        print(f"resumed from {args.resume}  start_epoch {start_epoch}  best_epoch_loss {best_loss:.6f}")
        print("run stamp:", run_stamp)

    latest_path = os.path.join(args.out_dir, f"latest_{run_stamp}.pt")
    best_path = os.path.join(args.out_dir, f"student_best_{run_stamp}.pt")

    for epoch in range(start_epoch, args.epochs + 1):
        if epoch <= args.freeze_epochs:
            freeze_by_name(model, args.trainable_names.split(","))
            print("freeze mode on. trainable substrings:", args.trainable_names)
        else:
            for p in model.parameters():
                p.requires_grad = True
            print("freeze mode off. training all parameters")

        model.train()
        running = 0.0
        steps = 0

        for step, batch in enumerate(loader):
            if want_teacher:
                tiles, meta, idx_list, prob_list, teacher_blobs = batch
            else:
                tiles, meta, idx_list, prob_list = batch
                teacher_blobs = None

            tiles = tiles.to(device, non_blocking=True)
            meta = meta.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(tiles, metadata=meta)

                loss_pseudo = masked_selftrain_loss(
                    logits,
                    idx_list=idx_list,
                    prob_list=prob_list,
                    neg_sample_k=args.neg_sample_k,
                )

                probs = torch.sigmoid(logits)
                loss_co = co_graph_loss_from_probs(
                    probs,
                    co_ei,
                    co_ej,
                    co_ew,
                    max_edges=args.co_edges_per_batch,
                )

                if want_teacher:
                    loss_teacher = masked_teacher_prior_loss(
                        logits,
                        teacher_blobs=teacher_blobs,
                        num_classes=args.num_classes,
                        teacher_conf_thr=args.teacher_conf_thr,
                        neg_sample_k=args.teacher_neg_sample_k,
                    )
                else:
                    loss_teacher = loss_pseudo * 0.0

                loss = loss_pseudo + (args.lambda_co * loss_co) + (args.lambda_teacher * loss_teacher)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            running += float(loss.item())
            steps += 1

            if (step + 1) % 50 == 0:
                print(f"  step {step+1:6d}  loss {running/max(1, steps):.6f}")

        epoch_loss = running / max(1, steps)
        print(f"epoch {epoch} loss {epoch_loss:.6f}")

        save_latest_checkpoint(
            latest_path,
            epoch=epoch,
            run_stamp=run_stamp,
            model=model,
            optimizer=optimizer,
            epoch_loss=epoch_loss,
            best_loss=best_loss,
            args=args,
        )
        print("saved latest checkpoint:", latest_path)

        if epoch_loss + 1e-12 < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), best_path)
            print("saved best:", best_path)


if __name__ == "__main__":
    main()



'''

python src/step3_v3/train_step3b2_selftrain_refinement.py \
  --merged-root /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/parquet/lucas_212k/student_ready_merged_input_step3a \
  --pseudo-parquet /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/parquet/lucas_212k/step3b_pseudolabels/top8_rel085.parquet \
  --init-ckpt /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/checkpoints/lucas_212k_step3a_semisup_cograph/student_best_YYYYMMDD_HHMM.pt \
  --out-dir /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/checkpoints/lucas_212k_step3b2_selftrain_from_semisup \
  --epochs 4 \
  --batch-size 64 \
  --num-workers 6 \
  --lr 3e-5 \
  --freeze-epochs 2 \
  --co-graph-npz /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/graphs/lucas_212k_cograph_ecology_top8_alpha060_pmi_min100_top50.npz \
  --lambda-co 0.01 \
  --co-edges-per-batch 200000 \
  --resume /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/checkpoints/lucas_212k_step3b2_selftrain_from_semisup/latest_YYYYMMDD_HHMM.pt

'''
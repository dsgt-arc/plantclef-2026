from pathlib import Path
import os

def plantclef_root() -> Path:
    return Path(os.environ["PLANTCLEF_ROOT"]).expanduser().resolve()

def lucas_raw_dir() -> Path:
    return plantclef_root() / "data" / "raw" / "lucas_10k_images"

def model_path() -> Path:
    return plantclef_root() / "pretrained_models" / "base_vit_model" / "vit_base_patch14_reg4_dinov2.pth"

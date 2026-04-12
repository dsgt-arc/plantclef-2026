from pathlib import Path
import os


def plantclef_root() -> Path:
    return Path(os.environ["PLANTCLEF_ROOT"]).expanduser().resolve()


def lucas_sw77k_metadata_csv() -> Path:
    return (
        plantclef_root()
        / "data"
        / "raw"
        / "metadata_lucas"
        / "LUCAS_metadata_SWE_Wide_lat35_50_lon-10_10_77k.csv"
    )


def lucas_images_dir() -> Path:
    return plantclef_root() / "data" / "raw" / "lucas_212k_images"


def test_images_dir() -> Path:
    return plantclef_root() / "data" / "raw" / "PlanCLEF_test_data" / "images"


def checkpoint_path() -> Path:
    return (
        plantclef_root()
        / "pretrained_models"
        / "base_vit_model"
        / "vit_base_patch14_reg4_dinov2.pth"
    )


def species_mask_path() -> Path:
    return (
        plantclef_root()
        / "aerten3"
        / "species_masking"
        / "sw_europe_mask_bool.npy"
    )


def class_mapping_path() -> Path:
    return (
        plantclef_root()
        / "data"
        / "raw"
        / "class_id_to_species_mapping"
        / "class_to_species_id.json"
    )


def parquet_root() -> Path:
    return plantclef_root() / "aerten3" / "parquet" / "lucas_sw77k"


def manifests_dir() -> Path:
    return plantclef_root() / "manifests"


def checkpoints_dir() -> Path:
    return plantclef_root() / "aerten3" / "checkpoints"


def submissions_dir() -> Path:
    return plantclef_root() / "aerten3" / "submissions"


def model_path() -> Path:
    return checkpoint_path()

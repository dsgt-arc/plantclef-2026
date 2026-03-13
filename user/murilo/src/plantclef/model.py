"""DINOv2 model loading and class mapping utilities."""

import timm
import torch


def load_model(
    checkpoint_path: str,
    model_name: str = "vit_base_patch14_reg4_dinov2.lvd142m",
    num_classes: int = 7806,
):
    """Load fine-tuned DINOv2 model and return (model, transforms).

    Args:
        checkpoint_path: Path to .pth or .pth.tar checkpoint.
        model_name: timm model identifier.
        num_classes: Number of output classes (7806 PlantCLEF species).

    Returns:
        Tuple of (model, transforms) where transforms is the timm data transform.
    """
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
        checkpoint_path=checkpoint_path,
    )
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    return model, transforms


def load_class_mapping(class_mapping_path: str) -> dict[int, int]:
    """Load class index -> species ID mapping from class_mapping.txt.

    Each line i in the file contains the species ID for class index i.

    Returns:
        Dict mapping class index (0-7805) to species ID (int).
    """
    with open(class_mapping_path) as f:
        return {i: int(line.strip()) for i, line in enumerate(f)}

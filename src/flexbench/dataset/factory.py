from flexbench.dataset.base import DatasetConfig, MLPerfDataset
from flexbench.dataset.text import TextDataset
from flexbench.dataset.vision import VisionDataset

DATASET_REGISTRY = {
    "text": TextDataset,
    "vision": VisionDataset,
}


def create_dataset(
    task_type: str, dataset_config: DatasetConfig, **kwargs
) -> MLPerfDataset:
    """Create dataset instance based on task type."""
    if task_type not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported task type: {task_type}")

    dataset_class = DATASET_REGISTRY[task_type]
    supported_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k in dataset_class.__init__.__code__.co_varnames
    }

    return dataset_class(dataset_config=dataset_config, **supported_kwargs)

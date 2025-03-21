import base64
from io import BytesIO
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from flexbench.dataset.base import DatasetConfig, MLPerfDataset, ReferenceData
from flexbench.utils import get_logger

log = get_logger(__name__)

SYSTEM_MESSAGE = "You are an expert product description writer for Amazon."
TEMPLATE_PROMPT = """Create a Short Product description based on the provided ##PRODUCT NAME## and ##CATEGORY## and image.
Only return description. The description should be SEO optimized and for a better mobile search experience.

##PRODUCT NAME##: {product_name}
##CATEGORY##: {category}"""


class VisionDataset(MLPerfDataset):
    """Dataset for vision-based tasks."""

    def __init__(
        self,
        dataset_config: DatasetConfig,
        *,
        max_generated_tokens: int | None = None,
        max_image_size: int = 512,
        image_quality: int = 85,
        **kwargs,
    ) -> None:
        self.max_generated_tokens = max_generated_tokens
        self.max_image_size = max_image_size
        self.image_quality = image_quality
        super().__init__(dataset_config, **kwargs)

    def _process_image(self, image_array: np.ndarray | str | Path) -> str:
        """Process image into base64 string."""
        try:
            if isinstance(image_array, (str, Path)):
                image = Image.open(image_array)
            elif isinstance(image_array, np.ndarray):
                image = Image.fromarray(image_array)
            elif isinstance(image_array, Image.Image):
                image = image_array
            else:
                raise ValueError(f"Unsupported image type: {type(image_array)}")

            if image.mode != "RGB":
                image = image.convert("RGB")

            if max(image.size) > self.max_image_size:
                ratio = self.max_image_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            buffered = BytesIO()
            image.save(
                buffered, format="JPEG", quality=self.image_quality, optimize=True
            )
            buffered.seek(0)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/jpeg;base64,{img_str}"

        except Exception as e:
            log.error(f"Image processing error: {e}")
            raise

    def _format_sample(self, sample: dict) -> dict:
        """Format a raw sample into a chat message format."""
        try:
            image_data = self._process_image(sample[self.config.image_column])
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_MESSAGE}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": TEMPLATE_PROMPT.format(
                                product_name=sample[self.config.input_column],
                                category=sample["Category"].replace("›", " | ").strip(),
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": image_data}},
                    ],
                },
            ]
            return {"messages": messages}
        except Exception as e:
            log.warning(f"Failed to process sample: {e}")
            return None

    def load_from_huggingface(self, dataset_path: str, split: str = "train") -> None:
        """Load and format data from HuggingFace dataset."""
        log.info(f"Loading vision dataset from HuggingFace: {dataset_path} ({split})")
        dataset = load_dataset(dataset_path, split=split)
        self.raw_samples = list(dataset)
        formatted_samples = []

        for sample in tqdm(dataset, desc="Formatting samples"):
            formatted = self._format_sample(sample)
            if formatted:
                formatted_samples.append(formatted)

        self.samples = formatted_samples
        log.info(f"Loaded {len(self)} samples")

    def get_references(self) -> ReferenceData:
        """Get raw reference data for accuracy evaluation."""
        if not self.raw_samples:
            log.warning("No samples found for reference data")
            return ReferenceData([], [], [])

        return ReferenceData(
            references=[
                sample[self.config.output_column] for sample in self.raw_samples
            ],
            inputs=[sample[self.config.input_column] for sample in self.raw_samples],
            system_prompts=[SYSTEM_MESSAGE] * len(self.raw_samples),
        )

import textwrap

from tqdm import tqdm

from flexbench.config import DatasetConfig
from flexbench.dataset.base import MLPerfDataset
from flexbench.utils import get_logger

log = get_logger(__name__)

MODEL_CONFIGS = {
    "llama2": {
        "pattern": "llama2",
        "template": textwrap.dedent(
            """
            <s>[INST] <<SYS>>
            {system_prompt}
            <</SYS>>

            {user_message} [/INST]
            """
        ),
    },
    "llama3": {
        "pattern": "llama3",
        "template": textwrap.dedent(
            """
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>

            {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

            {user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
        ),
    },
    "deepseek": {
        "pattern": "deepseek",
        "template": textwrap.dedent(
            """
            {system_prompt}

            {user_message}
            """
        ),
    },
    "smollm": {
        "pattern": "smollm",
        "template": textwrap.dedent(
            """
            {system_prompt}

            Human: {user_message}

            Assistant:"""
        ),
    },
}


class TextDataset(MLPerfDataset):
    """Dataset for text-based tasks."""

    def __init__(
        self,
        dataset_config: DatasetConfig,
        *,
        model_path: str,
        max_generated_tokens: int | None = None,
        max_input_tokens: int | None = None,
        fixed_input_length: bool = False,
    ) -> None:
        self.model_path = model_path
        self.max_generated_tokens = max_generated_tokens
        self.max_input_tokens = max_input_tokens
        self.fixed_input_length = fixed_input_length
        self.model_type = self._get_model_type()
        self.tokenizer = None

        if max_input_tokens is not None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            self.tokenizer.pad_token_id = (
                self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
            )
            log.info(f"Created tokenizer for input processing (max {max_input_tokens} tokens)")
            log.debug(f"Using pad token ID: {self.tokenizer.pad_token_id}")
            if fixed_input_length:
                log.info(f"Using fixed input length of {max_input_tokens} tokens")

        super().__init__(dataset_config)

    def _get_model_type(self) -> str:
        """Determine model type from model path."""
        model_path = self.model_path.lower()

        for model_type, config in MODEL_CONFIGS.items():
            if config["pattern"] in model_path:
                log.info(f"Detected {model_type=} with chat template: {repr(config['template'])}")
                return model_type

        log.warning(
            f"Model type not found among {list(MODEL_CONFIGS.keys())}. Using SmolLM template."
        )
        return "smollm"

    def _apply_template(self, sample: dict) -> str:
        """Apply the appropriate template to a sample."""
        system_prompt = (
            sample.get(self.config.system_prompt_column, "")
            if self.config.system_prompt_column
            else "You are an AI assistant."
        )

        config = MODEL_CONFIGS.get(self.model_type, MODEL_CONFIGS["smollm"])
        return (
            config["template"]
            .format(
                system_prompt=system_prompt,
                user_message=sample[self.config.input_column],
            )
            .strip()
        )

    def _format_samples_batch(self, samples: list[dict]) -> list[str]:
        """Format multiple samples efficiently in batch."""
        if not samples:
            return []

        formatted_prompts = [
            self._apply_template(sample) for sample in tqdm(samples, desc="Applying templates")
        ]

        if not self.tokenizer or self.max_input_tokens is None:
            return formatted_prompts

        log.info(f"Tokenizing {len(samples)} samples in batch")
        batch_size = 64
        processed_prompts = []

        for i in tqdm(range(0, len(formatted_prompts), batch_size), desc="Processing batches"):
            batch = formatted_prompts[i : i + batch_size]

            if self.fixed_input_length:
                for prompt in batch:
                    tokens = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.max_input_tokens,
                        padding=False,
                    ).input_ids[0]

                    if len(tokens) < self.max_input_tokens:
                        import torch

                        padding_size = self.max_input_tokens - len(tokens)
                        padding = torch.tensor(
                            [self.tokenizer.pad_token_id] * padding_size,
                            dtype=tokens.dtype,
                        )
                        padded_ids = torch.cat([tokens, padding])
                        processed_prompts.append(
                            self.tokenizer.decode(padded_ids, skip_special_tokens=False)
                        )
                    else:
                        processed_prompts.append(
                            self.tokenizer.decode(tokens, skip_special_tokens=True)
                        )
            else:
                encoded = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    padding_side="right",
                    truncation=True,
                    max_length=self.max_input_tokens,
                )

                for input_ids in encoded.input_ids:
                    processed_prompts.append(
                        self.tokenizer.decode(input_ids, skip_special_tokens=True)
                    )

        return processed_prompts

import textwrap

from flexbench.dataset.base import DatasetConfig, MLPerfDataset
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
    ) -> None:
        self.model_path = model_path
        self.max_generated_tokens = max_generated_tokens
        self.model_type = self.get_model_type()
        super().__init__(dataset_config)

    def get_model_type(self) -> str:
        """Determine model type from model path."""
        model_path = self.model_path.lower()

        # Check for SmolLM first
        if "smollm" in model_path:
            log.info("Detected SmolLM model, using SmolLM template")
            return "smollm"

        # Then check other models
        for model_type, config in MODEL_CONFIGS.items():
            if config["pattern"] in model_path:
                log.info(
                    f"Detected model type: {model_type} with chat template: {repr(config['template'])}"
                )
                return model_type

        # Default to SmolLM for instruction models
        log.warning(
            f"Model type not found among {list(MODEL_CONFIGS.keys())}. "
            "Using SmolLM template for instruction model."
        )
        return "smollm"

    def _format_sample(self, sample: dict) -> str:
        """Format a sample using the appropriate template."""
        system_prompt = (
            sample.get(self.config.system_prompt_column, "")
            if self.config.system_prompt_column
            else "You are an AI assistant that helps people find information."
        )

        config = MODEL_CONFIGS.get(self.model_type, MODEL_CONFIGS["smollm"])
        template = config["template"]

        return template.format(
            system_prompt=system_prompt,
            user_message=sample[self.config.input_column],
        ).strip()

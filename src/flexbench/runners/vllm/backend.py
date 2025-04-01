import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp

from flexbench.configs import BenchmarkConfig
from flexbench.runners.base import BaseBackend
from flexbench.utils import get_logger


@dataclass
class RequestOutput:
    """Results from a single request."""

    success: bool
    prompt_len: int
    ttft: float = 0.0  # Time to first token
    latency: float = 0.0  # Total request latency
    output_tokens: int = 0
    itl: list[float] = None  # Inter-token latencies
    generated_text: str = ""
    error: str | None = None
    batch_size: int = 1  # Added for offline mode


log = get_logger(__name__)


class VLLMBackend(BaseBackend):
    """vLLM backend implementation."""

    def __init__(self, config: BenchmarkConfig, results_dir: Path):
        super().__init__(config)
        self.results_dir = results_dir
        self.api_url = f"{config.api_server}/v1/completions"

    def start(self):
        self._active = True 

    def stop(self):
        self._active = False

    def process_query(self, query: dict) -> dict:
        return asyncio.run(self._make_request(query))

    async def _make_request(self, query: dict) -> dict:
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_token}"
                if self.config.api_token
                else None,
            }

            payload = {
                "model": self.config.model_path,
                "prompt": query["prompt"],
                "max_tokens": query["max_tokens"],
                "temperature": 0,
                "stream": True,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url, headers=headers, json=payload
                ) as resp:
                    resp.raise_for_status()

                    start_time = time.perf_counter()
                    first_token_received = False
                    generated_text = ""
                    itl = []
                    last_token_time = start_time
                    output_tokens = 0

                    async for line in resp.content:
                        if not self._active or not line or b"[DONE]" in line:
                            continue

                        decoded = line.decode()
                        if not decoded.startswith("data:"):
                            continue

                        curr_time = time.perf_counter()
                        data = json.loads(decoded[6:])
                        token_text = data["choices"][0]["text"]

                        if token_text:
                            if not first_token_received:
                                ttft = curr_time - start_time
                                first_token_received = True
                            else:
                                itl.append(curr_time - last_token_time)

                            last_token_time = curr_time
                            generated_text += token_text
                            output_tokens += 1

                    return RequestOutput(
                        success=True,
                        prompt_len=len(self.tokenizer(query["prompt"]).input_ids),
                        ttft=ttft if first_token_received else None,
                        latency=time.perf_counter() - start_time,
                        output_tokens=output_tokens,
                        itl=itl,
                        generated_text=generated_text,
                    )

        except Exception as e:
            return RequestOutput(
                success=False,
                prompt_len=len(self.tokenizer(query["prompt"]).input_ids),
                error=str(e),
            )

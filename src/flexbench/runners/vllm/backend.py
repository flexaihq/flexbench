import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp

from flexbench.config import BenchmarkConfig
from flexbench.runners.base import BaseBackend
from flexbench.utils import get_logger


@dataclass
class RequestOutput:
    """Results from a single request."""

    success: bool
    prompt_len: int
    ttft: float = 0.0
    latency: float = 0.0
    output_tokens: int = 0
    itl: list[float] | None = None
    generated_text: str = ""
    error: str | None = None
    batch_size: int = 1


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

    async def run(self) -> tuple[list[RequestOutput], float]:
        """Run benchmark and collect metrics."""
        self.start()
        start_time = time.perf_counter()
        outputs: list[RequestOutput] = []

        log.info("Starting vLLM benchmark run")
        log.info(f"Scenario: {self.config.scenario}")
        log.info(f"Target QPS: {self.config.target_qps}")
        log.info(f"Total samples: {self.total_sample_count}")
        log.info(f"API URL: {self.api_url}")

        if self.config.scenario == "Offline":
            batch_size = self.config.batch_size or len(self.dataset)
            log.info(f"Running in Offline mode with batch size: {batch_size}")
            for i in range(0, self.total_sample_count, batch_size):
                batch = self.dataset.get_batch(
                    range(i, min(i + batch_size, self.total_sample_count))
                )
                async with aiohttp.ClientSession():
                    tasks = []
                    for sample in batch:
                        tasks.append(
                            self._make_request(
                                {
                                    "prompt": sample,
                                    "max_tokens": self.config.max_generated_tokens,
                                }
                            )
                        )
                    batch_outputs = await asyncio.gather(*tasks)
                    outputs.extend(batch_outputs)
                    for _ in batch_outputs:
                        self._update_counter()

        elif self.config.scenario == "Server":
            interval = 1.0 / self.config.target_qps
            log.info(f"Running in Server mode with {interval:.2f}s interval")

            tasks = []
            for i in range(self.total_sample_count):
                sample = self.dataset.get_sample(i)
                task = asyncio.create_task(
                    self._make_request(
                        {
                            "prompt": sample,
                            "max_tokens": self.config.max_generated_tokens,
                        }
                    )
                )
                tasks.append(task)
                await asyncio.sleep(interval)

            outputs.extend(await asyncio.gather(*tasks))
            for _ in outputs:
                self._update_counter()

        else:
            raise ValueError(f"Unknown scenario: {self.config.benchmarking_config.scenario}")

        duration = time.perf_counter() - start_time
        log.info(f"Benchmark completed in {duration:.2f}s")
        log.info(f"Successful requests: {len([o for o in outputs if o.success])}/{len(outputs)}")
        self.stop()
        return outputs, duration

    async def _make_request(self, query: dict) -> RequestOutput:
        """Make API request and format output."""
        prompt = query["prompt"]
        start_time = time.perf_counter()

        headers = {"Content-Type": "application/json"}
        if self.config.api_token:
            headers["Authorization"] = f"Bearer {self.config.api_token}"

        payload = {
            "model": self.config.remote_model_path,
            "prompt": prompt,
            "max_tokens": query["max_tokens"],
            "temperature": 0,
            "stream": True,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, json=payload) as resp:
                resp.raise_for_status()

                start_time = time.perf_counter()
                first_token_received = False
                generated_text = ""
                itl = []
                last_token_time = start_time
                output_tokens = 0
                ttft = 0.0

                raw_content = ""
                async for chunk in resp.content:
                    raw_content += chunk.decode("utf-8")
                    if not self._active:
                        break

                    chunk_str = chunk.decode("utf-8")
                    if not chunk_str.strip():
                        continue

                    for line in chunk_str.split("\n"):
                        line = line.strip()
                        if not line:
                            continue

                        if line == "data: [DONE]":
                            continue

                        if line.startswith("data: "):
                            data = json.loads(line[6:])
                            if (
                                "choices" in data
                                and len(data["choices"]) > 0
                                and data["choices"][0].get("text")
                            ):
                                token_text = data["choices"][0]["text"]

                                curr_time = time.perf_counter()
                                if token_text:
                                    if not first_token_received:
                                        ttft = curr_time - start_time
                                        first_token_received = True
                                        last_token_time = curr_time
                                    else:
                                        itl.append(curr_time - last_token_time)
                                        last_token_time = curr_time

                                    generated_text += token_text
                                    output_tokens += 1

                latency = time.perf_counter() - start_time
                success = output_tokens > 0

                if not success:
                    log.warning("Request completed but no tokens were generated.")
                    log.warning(f"Raw content: {raw_content}")
                    log.warning(f"Response: {resp.status} {resp.reason}")
                    log.warning(f"Payload: {payload}")
                    log.warning(f"Headers: {headers}")

                return RequestOutput(
                    success=success,
                    prompt_len=len(self.tokenizer(query["prompt"]).input_ids),
                    ttft=ttft,
                    latency=latency,
                    output_tokens=output_tokens,
                    itl=itl,
                    generated_text=generated_text,
                    batch_size=getattr(self.config, "batch_size", 1),
                )

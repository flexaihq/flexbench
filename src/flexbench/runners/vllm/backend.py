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

    async def run(self) -> tuple[list[RequestOutput], float]:
        """Run benchmark and collect metrics."""
        try:
            self.start()
            start_time = time.perf_counter()
            outputs: list[RequestOutput] = []

            log.info("Starting vLLM benchmark run")
            log.info(f"Scenario: {self.config.benchmarking_config.scenario}")
            log.info(f"Target QPS: {self.config.benchmarking_config.target_qps}")
            log.info(f"Total samples: {self.total_sample_count}")
            log.info(f"API URL: {self.api_url}")

            # Offline mode processes all samples in batches
            if self.config.benchmarking_config.scenario == "Offline":
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

            # Server mode processes samples concurrently at target QPS rate
            else:
                interval = 1.0 / self.config.benchmarking_config.target_qps
                log.info(f"Running in Server mode with {interval:.2f}s interval")

                # Create tasks at the target QPS rate
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

                # Wait for all tasks to complete
                outputs.extend(await asyncio.gather(*tasks))
                for _ in outputs:
                    self._update_counter()

            duration = time.perf_counter() - start_time
            log.info(f"Benchmark completed in {duration:.2f}s")
            log.info(
                f"Successful requests: {len([o for o in outputs if o.success])}/{len(outputs)}"
            )
            return outputs, duration

        except Exception as e:
            log.error(f"Benchmark run failed: {str(e)}")
            raise
        finally:
            self.stop()

    async def _make_request(self, query: dict) -> RequestOutput:
        """Make API request and format output."""
        try:
            prompt = query["prompt"]
            log.debug(f"Sending request with {len(prompt)} chars")
            log.debug(f"Full prompt text:\n{prompt}")
            start_time = time.perf_counter()

            # Fix headers - only add Authorization if token exists
            headers = {"Content-Type": "application/json"}
            if self.config.api_token:
                headers["Authorization"] = f"Bearer {self.config.api_token}"

            payload = {
                "model": self.config.model_path,
                "prompt": prompt,
                "max_tokens": query["max_tokens"],
                "temperature": 0,
                "stream": True,
            }

            log.debug(f"Request payload: {json.dumps(payload, indent=2)}")

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url, headers=headers, json=payload
                ) as resp:
                    resp.raise_for_status()
                    log.debug(f"Request sent successfully to {self.api_url}")
                    log.debug(f"Response status: {resp.status}")
                    log.debug(f"Response headers: {dict(resp.headers)}")

                    start_time = time.perf_counter()
                    first_token_received = False
                    generated_text = ""
                    itl = []
                    last_token_time = start_time
                    output_tokens = 0
                    ttft = None

                    # Debug raw response line by line
                    raw_content = ""
                    async for chunk in resp.content:
                        raw_content += chunk.decode("utf-8")
                        if not self._active:
                            break

                        try:
                            chunk_str = chunk.decode("utf-8")
                            log.debug(
                                f"Processing chunk: {chunk_str}"
                            )  # Added debug logging
                            if not chunk_str.strip():
                                continue

                            for line in chunk_str.split("\n"):
                                line = line.strip()
                                if not line:
                                    continue

                                if line == "data: [DONE]":
                                    continue

                                if line.startswith("data: "):
                                    try:
                                        data = json.loads(line[6:])
                                        log.debug(
                                            f"Parsed response data: {json.dumps(data, indent=2)}"
                                        )  # Added debug
                                        if (
                                            "choices" in data
                                            and len(data["choices"]) > 0
                                            and data["choices"][0].get(
                                                "text"
                                            )  # Check if text exists and not empty
                                        ):
                                            token_text = data["choices"][0]["text"]

                                            curr_time = time.perf_counter()
                                            if token_text:
                                                if not first_token_received:
                                                    ttft = curr_time - start_time
                                                    first_token_received = True
                                                    last_token_time = curr_time
                                                    log.debug(
                                                        f"First token received in {ttft:.3f}s"
                                                    )
                                                else:
                                                    itl.append(
                                                        curr_time - last_token_time
                                                    )
                                                    last_token_time = curr_time

                                                generated_text += token_text
                                                output_tokens += 1

                                    except json.JSONDecodeError as e:
                                        log.warning(
                                            f"Failed to parse JSON: {line} - {str(e)}"
                                        )
                                        continue

                        except Exception as e:
                            log.warning(f"Error processing chunk: {str(e)}")
                            continue

                    latency = time.perf_counter() - start_time
                    success = output_tokens > 0

                    if not success:
                        log.warning(
                            f"Request completed but no tokens were generated.\n"
                            f"Full prompt:\n{prompt}\n"
                            f"Raw response:\n{raw_content}\n"
                            f"Check model configuration and ensure correct chat template is used."  # Added hint
                        )
                    else:
                        log.debug(f"Generated text:\n{generated_text}")
                        log.debug(f"Generated {output_tokens} tokens in {latency:.3f}s")

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

        except aiohttp.ClientError as e:
            log.error(f"API request error: {str(e)}")
            return RequestOutput(
                success=False,
                prompt_len=len(self.tokenizer(query["prompt"]).input_ids),
                error=str(e),
            )
        except json.JSONDecodeError as e:
            log.error(f"JSON decode error: {str(e)}")
            return RequestOutput(
                success=False,
                prompt_len=len(self.tokenizer(query["prompt"]).input_ids),
                error=f"JSON decode error: {str(e)}",
            )
        except Exception as e:
            log.error(f"Unexpected error: {str(e)}")
            return RequestOutput(
                success=False,
                prompt_len=len(self.tokenizer(query["prompt"]).input_ids),
                error=str(e),
            )

import array
import json
import os
import queue
import threading
from abc import ABC, abstractmethod
from math import log10

import mlperf_loadgen as lg
import numpy as np
import requests
import urllib3
from transformers import AutoTokenizer

from flexbench.configs import LoadgenConfig
from flexbench.dataset.base import DatasetConfig
from flexbench.dataset.factory import create_dataset
from flexbench.utils import get_logger

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

log = get_logger(__name__)


class SUT(ABC):
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str | None,
        api_server: str,
        api_token: str | None,
        dataset_config: DatasetConfig,
        loadgen_config: LoadgenConfig,
        task_type: str = "text",
        max_generated_tokens: int | None = None,
        **kwargs,
    ):
        log.info(f"Initializing SUT with:\n{locals()}")
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.api_server = api_server
        self.api_token = api_token
        self.dataset_config = dataset_config
        self.loadgen_config = loadgen_config
        self.task_type = task_type

        self.dataset = create_dataset(
            task_type,
            dataset_config,
            model_path=model_path,
            max_generated_tokens=max_generated_tokens,
            **kwargs,
        )

        self.total_sample_count = loadgen_config.total_sample_count or len(self.dataset)
        self.perf_sample_count = self.total_sample_count

        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

        if task_type == "text":
            self._init_text_task(max_generated_tokens)
        elif task_type == "vision":
            self._init_vision_task(max_generated_tokens)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    def _init_text_task(self, max_generated_tokens: int | None = None) -> None:
        """Initialize for text generation tasks."""
        log.info("Initializing text task")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            use_fast=True,
            add_prefix_space=None if self.dataset.model_type == "deepseek" else False,
            padding_size="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_tokens = min(
            self.tokenizer.model_max_length,
            max_generated_tokens or float("inf"),
        )

    def _init_vision_task(self, max_generated_tokens: int | None = None) -> None:
        """Initialize for vision tasks."""
        log.info("Initializing vision task")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            use_fast=True,
        )
        self.max_tokens = min(
            self.tokenizer.model_max_length,
            max_generated_tokens or float("inf"),
        )

    def _create_api_payload(
        self, inputs: str | dict | list, stream: bool = False
    ) -> tuple[str, dict]:
        """Create API payload and endpoint based on task type and streaming mode."""
        base_config = {
            "model": self.model_path,
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "stream": stream,
        }

        if self.task_type == "text":
            endpoint = f"{self.api_server}/v1/completions"
            payload = {
                **base_config,
                "prompt": inputs,
                "min_tokens": 1,
            }

        elif self.task_type == "vision":
            endpoint = f"{self.api_server}/v1/chat/completions"
            messages = (
                [inp["messages"] for inp in inputs]
                if isinstance(inputs, list)
                else inputs["messages"]
            )
            payload = {**base_config, "messages": messages}

        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        return endpoint, payload

    def _get_completion_text(
        self, response: dict, streaming: bool = False
    ) -> str | list[str]:
        """Extract completion text from API response."""
        if self.task_type == "text":
            if streaming:
                return response["choices"][0]["text"]
            return [choice["text"] for choice in response["choices"]]
        elif self.task_type == "vision":
            if streaming:
                return response["choices"][0]["delta"]["content"]
            return response["choices"][0]["message"]["content"]

    def _make_api_request(
        self, inputs: str | dict | list, stream: bool = False
    ) -> dict | requests.Response:
        """Make API request with proper headers and error handling."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_token}" if self.api_token else None,
        }
        endpoint, payload = self._create_api_payload(inputs, stream)

        with requests.Session() as s:
            resp = s.post(
                endpoint, headers=headers, json=payload, verify=False, stream=stream
            )
            try:
                resp.raise_for_status()
            except:
                log.error(f"API request failed: {resp.text}")
                log.error(f"Payload: {json.dumps(payload, indent=2)}")
                raise ValueError(f"API Error: {resp.text}")
            return resp if stream else resp.json()

    def _process_response(self, response: dict | str, streaming: bool = False) -> str:
        """Process API response and extract text content."""
        if isinstance(response, str):
            return response

        if self.task_type == "text":
            if streaming:
                return response["choices"][0]["text"]
            return "".join(choice["text"] for choice in response["choices"])
        elif self.task_type == "vision":
            if streaming:
                return response["choices"][0]["delta"]["content"]
            return response["choices"][0]["message"]["content"]

    def _process_completion(
        self, text: str, query_id: int, is_first_token: bool = False
    ) -> None:
        """Process completion text and submit response."""
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not token_ids and not is_first_token:
            log.warning(f"No output tokens generated for query {query_id}")
        self.submit_lg_response(token_ids, query_id, first_token=is_first_token)

    @abstractmethod
    def start(self) -> None:
        """Start the SUT processing. This method should be overridden."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop the SUT and clean up resources. This method should be overridden."""
        ...

    @abstractmethod
    def issue_queries(self, query_samples: list[lg.QuerySample]) -> None:
        """Issue queries to the SUT. This method should be overridden."""
        ...

    def submit_lg_response(
        self, token_ids: list[int], query_id: int, first_token: bool = False
    ) -> None:
        """Submit token response to MLPerf loadgen."""
        tokens_arr = np.array(token_ids, dtype=np.int32)
        resp_arr = array.array("B", tokens_arr.tobytes())
        bi = resp_arr.buffer_info()
        response = lg.QuerySampleResponse(query_id, bi[0], bi[1], len(tokens_arr))
        if first_token:
            lg.FirstTokenComplete([response])
        else:
            lg.QuerySamplesComplete([response])

    def log_progress(self, sample_counter: int) -> None:
        """Log current progress of sample processing."""
        percent = sample_counter / self.total_sample_count * 100
        log_interval = 10 ** int(log10(self.total_sample_count - 1) - 1)

        if (
            self.total_sample_count < 100
            or sample_counter == 1
            or sample_counter % log_interval == 0
            or sample_counter == self.total_sample_count
        ):
            log.info(
                f"Progress: {sample_counter}/{self.total_sample_count} samples "
                f"({percent:.1f}%)"
            )

    @property
    def sut(self):
        """Return MLPerf loadgen System Under Test object."""
        return lg.ConstructSUT(self.issue_queries, self.flush_queries)

    @property
    def qsl(self):
        """Return MLPerf loadgen Query Sample Library object."""
        return lg.ConstructQSL(
            self.total_sample_count,
            self.perf_sample_count,
            self.dataset.LoadSamplesToRam,
            self.dataset.UnloadSamplesFromRam,
        )

    def flush_queries(self):
        """Flush pending queries (required by MLPerf loadgen)."""
        pass


class SUTOffline(SUT):
    def __init__(self, batch_size: int | None = None, **kwargs):
        """Initialize offline SUT with optional batch size."""
        super().__init__(**kwargs)
        self.batch_size = batch_size or self.total_sample_count
        self.worker_threads: list[threading.Thread] = []
        self.query_queue = queue.Queue()

    def start(self) -> None:
        """Start worker threads for offline batch processing."""
        log.info("Starting SUT offline mode processing threads")
        num_workers = os.cpu_count()
        for _ in range(num_workers):
            worker = threading.Thread(target=self.process_queries)
            worker.start()
            self.worker_threads.append(worker)

    def stop(self) -> None:
        """Stop all worker threads and clean up resources."""
        log.info("Stopping SUT and cleaning up resources")
        for _ in range(len(self.worker_threads)):
            self.query_queue.put(None)

        for thread in self.worker_threads:
            if thread and thread.is_alive():
                thread.join()

    def issue_queries(self, query_samples: lg.QuerySample) -> None:
        """Process query samples in batches using worker threads."""
        if not self.worker_threads:
            self.start()

        while len(query_samples) > 0:
            batch = query_samples[: self.batch_size]
            self.query_queue.put(batch)
            query_samples = query_samples[self.batch_size :]

    def process_queries(self) -> None:
        """Worker thread function to process batches from queue."""
        while True:
            batch = self.query_queue.get()
            if batch is None:
                break

            log.debug(f"Processing batch of {len(batch)} queries")
            inputs = [self.dataset.samples[q.index] for q in batch]

            if self.task_type == "vision":
                # Process all inputs sequentially instead of submitting batch
                # TODO: find how to submit whole batch at once with vLLM API
                for i, input_data in enumerate(inputs):
                    response = self._make_api_request(input_data, stream=False)
                    output_text = self._process_response(response)
                    self._process_completion(output_text, batch[i].id)
                    self._update_counter()
            elif self.task_type == "text":
                response = self._make_api_request(inputs, stream=False)
                outputs = self._get_completion_text(response)
                for i, output_text in enumerate(outputs):
                    self._process_completion(output_text, batch[i].id)
                    self._update_counter()

    def _update_counter(self):
        """Update and log sample counter."""
        with self.sample_counter_lock:
            self.sample_counter += 1
            self.log_progress(self.sample_counter)


class SUTServer(SUT):
    def __init__(self, **kwargs):
        """Initialize server SUT for streaming responses."""
        super().__init__(**kwargs)
        self.first_token_queue = queue.Queue()

    def start(self) -> None:
        """Start first token processing thread."""
        log.info("Starting SUT server mode processing threads")
        self.ft_resp_thread = threading.Thread(target=self.process_first_tokens)
        self.ft_resp_thread.start()

    def stop(self) -> None:
        """Stop first token processing thread."""
        if hasattr(self, "ft_resp_thread"):
            self.first_token_queue.put(None)
            self.ft_resp_thread.join()

    def issue_queries(self, query_samples: list[lg.QuerySample]) -> None:
        """Process queries individually in separate threads."""
        if not hasattr(self, "ft_resp_thread"):
            self.start()

        for sample in query_samples:
            threading.Thread(target=self.process_query, args=(sample,)).start()

    def process_query(self, query_sample: lg.QuerySample) -> None:
        """Process a single query with streaming response."""
        input_data = self.dataset.samples[query_sample.index]
        response = self._make_api_request(input_data, stream=True)
        text_cache = ""
        first_token_sent = False

        for line in response.iter_lines():
            if not line or b"[DONE]" in line:
                continue

            decoded = line.decode()
            if not decoded.startswith("data"):
                continue

            token_data = json.loads(decoded[6:])
            token_text = self._process_response(token_data, streaming=True)

            if not token_text:
                continue

            if not first_token_sent:
                self._process_completion(
                    token_text, query_sample.id, is_first_token=True
                )
                first_token_sent = True

            text_cache += token_text

        self._process_completion(text_cache, query_sample.id)
        self._update_counter()

    def process_first_tokens(self) -> None:
        """Process and submit first tokens from streaming responses."""
        while True:
            item = self.first_token_queue.get()
            if item is None:
                break
            first_token_txt, query_id = item
            first_token_id = self.tokenizer.encode(
                first_token_txt,
                add_special_tokens=False,
            )
            self.submit_lg_response(first_token_id, query_id, first_token=True)

    def _update_counter(self):
        """Update and log sample counter."""
        with self.sample_counter_lock:
            self.sample_counter += 1
            self.log_progress(self.sample_counter)

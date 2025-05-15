import array
import json
import os
import queue
import threading
from pathlib import Path

import mlperf_loadgen as lg
import numpy as np
import urllib3

from flexbench.runners.base import BaseBackend, BenchmarkConfig
from flexbench.utils import get_logger

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

log = get_logger(__name__)


class LoadGenBackend(BaseBackend):
    """MLPerf LoadGen backend implementation."""

    def __init__(self, config: BenchmarkConfig, results_dir: Path):
        super().__init__(config)
        self.results_dir = results_dir
        self.task_type = config.task
        self.scenario = config.scenario

        if self.scenario == "SingleStream":
            self.batch_size = config.batch_size or 1
        elif self.scenario == "Offline":
            self.batch_size = config.batch_size or self.total_sample_count
        elif self.scenario == "Server":
            self.batch_size = config.batch_size

        self.qsl = lg.ConstructQSL(
            self.total_sample_count,
            self.total_sample_count,
            self.dataset.LoadSamplesToRam,
            self.dataset.UnloadSamplesFromRam,
        )
        log.info(f"Constructed QSL with {self.total_sample_count} samples")
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        log.info("Constructed SUT")

        if self.scenario == "Server":
            self.first_token_queue = queue.Queue()
            self.ft_resp_thread = None
        elif self.scenario == "Offline":
            self.worker_threads: list[threading.Thread] = []
            self.query_queue = queue.Queue()
        elif self.scenario == "SingleStream":
            pass

    def start(self):
        """Start the backend based on scenario type."""
        super().start()
        if self.scenario == "Server":
            self._start_server_scenario()
        elif self.scenario == "Offline":
            self._start_offline_scenario()
        elif self.scenario == "SingleStream":
            pass

    def stop(self):
        """Stop the backend and clean up resources."""
        if self.scenario == "Server":
            self._stop_server_scenario()
        elif self.scenario == "Offline":
            self._stop_offline_scenario()
        elif self.scenario == "SingleStream":
            pass

        if self.sut:
            lg.DestroySUT(self.sut)
        lg.DestroyQSL(self.qsl)
        super().stop()

    def process_query(self, query: dict) -> dict:
        """Process a query directly or queue it based on scenario."""
        return query

    def issue_queries(self, query_samples: list[lg.QuerySample]) -> None:
        """Issue queries according to the chosen scenario."""
        if self.scenario == "Server":
            self._issue_server_queries(query_samples)
        elif self.scenario == "Offline":
            self._issue_offline_queries(query_samples)
        elif self.scenario == "SingleStream":
            self._issue_singlestream_queries(query_samples)
        else:
            raise NotImplementedError(f"Scenario '{self.scenario}' not implemented.")

    def flush_queries(self):
        """Flush any pending queries."""
        pass

    # --- Shared helpers ---

    def _handle_streaming_response(self, query_sample, input_data):
        """Shared streaming response logic for Server and SingleStream."""
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
                self.process_completion(
                    token_text, query_sample.id, is_first_token=True
                )
                first_token_sent = True
            text_cache += token_text

        self.process_completion(text_cache, query_sample.id)
        self._update_counter()

    def _handle_batch_response(self, batch, inputs):
        """Shared batch response logic for Offline and SingleStream (non-stream)."""
        response = self._make_api_request(inputs, stream=False)
        outputs = response["choices"]
        for i, output in enumerate(outputs):
            output_text = output["text"]
            self.process_completion(output_text, batch[i].id)
            self._update_counter()

    def _process_first_tokens(self) -> None:
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
            self.submit_response(first_token_id, query_id, first_token=True)

    def submit_response(
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

    def process_completion(
        self, text: str, query_id: int, is_first_token: bool = False
    ) -> None:
        """Process completion text and submit response."""
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not token_ids and not is_first_token:
            log.warning(f"No output tokens generated for query {query_id}")
        self.submit_response(token_ids, query_id, first_token=is_first_token)

    # --- Scenario-specific methods ---

    def _start_offline_scenario(self) -> None:
        """Start worker threads for offline batch processing."""
        log.info("Starting SUT offline mode processing threads")
        num_workers = os.cpu_count()
        for _ in range(num_workers):
            worker = threading.Thread(target=self._process_offline_queries)
            worker.start()
            self.worker_threads.append(worker)

    def _stop_offline_scenario(self) -> None:
        """Stop all offline worker threads."""
        log.info("Stopping offline processing threads")
        for _ in range(len(self.worker_threads)):
            self.query_queue.put(None)
        for thread in self.worker_threads:
            if thread and thread.is_alive():
                thread.join()

    def _issue_offline_queries(self, query_samples: list[lg.QuerySample]) -> None:
        """Process query samples in batches using worker threads."""
        if not self.worker_threads:
            self._start_offline_scenario()
        for i in range(0, len(query_samples), self.batch_size):
            batch = query_samples[i : i + self.batch_size]
            self.query_queue.put(batch)

    def _process_offline_queries(self) -> None:
        """Worker thread function to process batches from queue."""
        while True:
            batch = self.query_queue.get()
            if batch is None:
                break
            log.debug(f"Processing batch of {len(batch)} queries")
            inputs = [self.dataset.get_sample(q.index) for q in batch]
            self._handle_batch_response(batch, inputs)

    def _start_server_scenario(self) -> None:
        """Start first token processing thread for server mode."""
        log.info("Starting SUT server mode processing thread")
        self.ft_resp_thread = threading.Thread(target=self._process_first_tokens)
        self.ft_resp_thread.start()

    def _stop_server_scenario(self) -> None:
        """Stop first token processing thread."""
        if hasattr(self, "ft_resp_thread") and self.ft_resp_thread:
            self.first_token_queue.put(None)
            self.ft_resp_thread.join()

    def _issue_server_queries(self, query_samples: list[lg.QuerySample]) -> None:
        """Process server queries individually in separate threads."""
        for sample in query_samples:
            threading.Thread(
                target=self._handle_streaming_response,
                args=(sample, self.dataset.get_sample(sample.index)),
            ).start()

    def _issue_singlestream_queries(self, query_samples: list[lg.QuerySample]) -> None:
        """Process queries in batches for SingleStream scenario."""
        for i in range(0, len(query_samples), self.batch_size):
            batch = query_samples[i : i + self.batch_size]
            for sample in batch:
                self._process_singlestream_query(sample)

    def _process_singlestream_query(self, query_sample: lg.QuerySample) -> None:
        """Process a single SingleStream query and submit first/full token responses."""
        input_data = self.dataset.get_sample(query_sample.index)
        self._handle_streaming_response(query_sample, input_data)

import array
import json
import os
import queue
import threading
from pathlib import Path

import mlperf_loadgen as lg
import numpy as np
import urllib3

from flexbench.configs import BenchmarkConfig
from flexbench.runners.base import BaseBackend
from flexbench.utils import get_logger

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

log = get_logger(__name__)


class LoadGenBackend(BaseBackend):
    """MLPerf LoadGen backend implementation."""

    def __init__(self, config: BenchmarkConfig, results_dir: Path):
        super().__init__(config)
        self.results_dir = results_dir
        self.task_type = config.task

        # Initialize MLPerf QSL once for both scenarios
        self._qsl = lg.ConstructQSL(
            self.total_sample_count,
            self.total_sample_count,
            self.dataset.LoadSamplesToRam,
            self.dataset.UnloadSamplesFromRam,
        )

        # Create appropriate scenario backend
        self._scenario = (
            LoadGenOfflineBackend
            if config.benchmarking_config.scenario == "Offline"
            else LoadGenServerBackend
        )(config=config, parent=self)

    def start(self):
        self._scenario.start()

    def stop(self):
        self._scenario.stop()
        lg.DestroySUT(self._scenario.sut)
        lg.DestroyQSL(self._qsl)

    def process_query(self, query: dict) -> dict:
        return self._scenario.process_query(query)

    @property
    def sut(self):
        return self._scenario.sut

    @property
    def qsl(self):
        return self._qsl

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

    def process_completion(
        self, text: str, query_id: int, is_first_token: bool = False
    ) -> None:
        """Process completion text and submit response."""
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not token_ids and not is_first_token:
            log.warning(f"No output tokens generated for query {query_id}")
        self.submit_lg_response(token_ids, query_id, first_token=is_first_token)


class LoadGenOfflineBackend:
    """LoadGen offline scenario backend."""

    def __init__(self, config: BenchmarkConfig, parent: LoadGenBackend):
        self.config = config
        self.parent = parent
        self.batch_size = config.batch_size or parent.total_sample_count
        self.worker_threads: list[threading.Thread] = []
        self.query_queue = queue.Queue()

        # Initialize MLPerf SUT
        self._sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def flush_queries(self):
        pass

    @property
    def sut(self):
        return self._sut

    def process_query(self, query_sample: lg.QuerySample) -> None:
        """Process a single query sample (required for interface but not used)."""
        pass  # Offline mode uses batched processing instead

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

    def issue_queries(self, query_samples: list[lg.QuerySample]) -> None:
        """Process query samples in batches using worker threads."""
        if not self.worker_threads:
            self.start()

        for i in range(0, len(query_samples), self.batch_size):
            batch = query_samples[i : i + self.batch_size]
            self.query_queue.put(batch)

    def process_queries(self) -> None:
        """Worker thread function to process batches from queue."""
        while True:
            batch = self.query_queue.get()
            if batch is None:
                break

            log.debug(f"Processing batch of {len(batch)} queries")
            inputs = [self.parent.dataset.get_sample(q.index) for q in batch]
            response = self.parent._make_api_request(inputs, stream=False)
            outputs = response["choices"]

            for i, output in enumerate(outputs):
                output_text = output["text"]
                self.parent.process_completion(output_text, batch[i].id)
                self.parent._update_counter()


class LoadGenServerBackend:
    """LoadGen server scenario backend."""

    def __init__(self, config: BenchmarkConfig, parent: LoadGenBackend):
        self.config = config
        self.parent = parent
        self.dataset = parent.dataset  # Add dataset access from parent
        self.first_token_queue = queue.Queue()

        # Initialize MLPerf SUT
        self._sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def flush_queries(self):
        pass

    @property
    def sut(self):
        return self._sut

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
        input_data = self.dataset.get_sample(query_sample.index)
        response = self.parent._make_api_request(input_data, stream=True)
        text_cache = ""
        first_token_sent = False

        for line in response.iter_lines():
            if not line or b"[DONE]" in line:
                continue

            decoded = line.decode()
            if not decoded.startswith("data"):
                continue

            token_data = json.loads(decoded[6:])
            token_text = self.parent._process_response(
                token_data, streaming=True
            )  # Use parent's method

            if not token_text:
                continue

            if not first_token_sent:
                self.parent.process_completion(
                    token_text, query_sample.id, is_first_token=True
                )
                first_token_sent = True

            text_cache += token_text

        self.parent.process_completion(text_cache, query_sample.id)
        self.parent._update_counter()  # Use parent's counter update

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
            self.parent.submit_lg_response(first_token_id, query_id, first_token=True)

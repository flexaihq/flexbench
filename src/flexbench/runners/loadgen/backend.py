import array
import json
import os
import queue
import threading
from pathlib import Path

import mlperf_loadgen as lg
import numpy as np
import urllib3

from flexbench.config import BenchmarkConfig
from flexbench.runners.base import BaseBackend
from flexbench.utils import get_logger

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

log = get_logger(__name__)


class SUT_Server:
    def __init__(self, backend: "LoadGenBackend"):
        self.backend = backend
        self.first_token_queue = queue.Queue()
        self.ft_resp_thread = None

    def start(self):
        log.info("Starting SUT server mode processing thread")
        self.ft_resp_thread = threading.Thread(target=self._process_first_tokens)
        self.ft_resp_thread.start()

    def stop(self):
        if self.ft_resp_thread:
            self.first_token_queue.put(None)
            self.ft_resp_thread.join()

    def issue_queries(self, query_samples: list[lg.QuerySample]) -> None:
        for sample in query_samples:
            threading.Thread(
                target=self._process_server_query,
                args=(sample,),
            ).start()

    def flush_queries(self):
        pass

    def _process_server_query(self, query_sample: lg.QuerySample) -> None:
        input_data = self.backend.dataset.get_sample(query_sample.index)
        response = self.backend._make_api_request(input_data, stream=True)
        text_cache = ""
        first_token_sent = False

        for line in response.iter_lines():
            if not line or b"[DONE]" in line:
                continue
            decoded = line.decode()
            if not decoded.startswith("data"):
                continue
            token_data = json.loads(decoded[6:])
            token_text = self.backend._process_response(token_data, streaming=True)
            if not token_text:
                continue
            if not first_token_sent:
                self.backend.process_completion(token_text, query_sample.id, is_first_token=True)
                first_token_sent = True
            text_cache += token_text

        self.backend.process_completion(text_cache, query_sample.id)
        self.backend._update_counter()

    def _process_first_tokens(self) -> None:
        while True:
            item = self.first_token_queue.get()
            if item is None:
                break
            first_token_txt, query_id = item
            first_token_id = self.backend.tokenizer.encode(
                first_token_txt,
                add_special_tokens=False,
            )
            self.backend.submit_response(first_token_id, query_id, first_token=True)


class SUT_Offline:
    def __init__(self, backend: "LoadGenBackend"):
        self.backend = backend
        self.worker_threads: list[threading.Thread] = []
        self.query_queue = queue.Queue()

    def start(self):
        log.info("Starting SUT offline mode processing threads")
        num_workers = os.cpu_count() or 0
        for _ in range(num_workers):
            worker = threading.Thread(target=self._process_offline_queries)
            worker.start()
            self.worker_threads.append(worker)

    def stop(self):
        log.info("Stopping offline processing threads")
        for _ in range(len(self.worker_threads)):
            self.query_queue.put(None)
        for thread in self.worker_threads:
            if thread and thread.is_alive():
                thread.join()

    def issue_queries(self, query_samples: list[lg.QuerySample]) -> None:
        if not self.worker_threads:
            self.start()
        for i in range(0, len(query_samples), self.backend.batch_size):
            batch = query_samples[i : i + self.backend.batch_size]
            self.query_queue.put(batch)

    def flush_queries(self):
        pass

    def _process_offline_queries(self) -> None:
        while True:
            batch = self.query_queue.get()
            if batch is None:
                break
            log.debug(f"Processing batch of {len(batch)} queries")
            inputs = [self.backend.dataset.get_sample(q.index) for q in batch]
            response = self.backend._make_api_request(inputs, stream=False)
            outputs = response["choices"]
            for i, output in enumerate(outputs):
                output_text = output["text"]
                self.backend.process_completion(output_text, batch[i].id)
                self.backend._update_counter()


class SUT_SingleStream:
    def __init__(self, backend: "LoadGenBackend"):
        self.backend = backend

    def start(self):
        pass

    def stop(self):
        pass

    def issue_queries(self, query_samples: list[lg.QuerySample]) -> None:
        for i in range(0, len(query_samples), self.backend.batch_size):
            batch = query_samples[i : i + self.backend.batch_size]
            for sample in batch:
                self._process_singlestream_query(sample)

    def flush_queries(self):
        pass

    def _process_singlestream_query(self, query_sample: lg.QuerySample) -> None:
        input_data = self.backend.dataset.get_sample(query_sample.index)
        response = self.backend._make_api_request(input_data, stream=True)
        text_cache = ""
        first_token_sent = False

        for line in response.iter_lines():
            if not line or b"[DONE]" in line:
                continue
            decoded = line.decode()
            if not decoded.startswith("data"):
                continue
            token_data = json.loads(decoded[6:])
            token_text = self.backend._process_response(token_data, streaming=True)
            if not token_text:
                continue
            if not first_token_sent:
                self.backend.process_completion(token_text, query_sample.id, is_first_token=True)
                first_token_sent = True
            text_cache += token_text

        self.backend.process_completion(text_cache, query_sample.id)
        self.backend._update_counter()


# --- LoadGenBackend ---


class LoadGenBackend(BaseBackend):
    """MLPerf LoadGen backend implementation."""

    def __init__(self, config: BenchmarkConfig, results_dir: Path):
        super().__init__(config)
        self.results_dir = results_dir
        self.task_type = "text"  # Only text tasks supported now
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

        if self.scenario == "Server":
            self.sut_impl = SUT_Server(self)
        elif self.scenario == "Offline":
            self.sut_impl = SUT_Offline(self)
        elif self.scenario == "SingleStream":
            self.sut_impl = SUT_SingleStream(self)
        else:
            raise ValueError(f"Unknown scenario: {self.scenario}")

        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        log.info("Constructed SUT")

    def start(self):
        """Start the backend based on scenario type."""
        super().start()
        self.sut_impl.start()

    def stop(self):
        """Stop the backend and clean up resources."""
        self.sut_impl.stop()
        if self.sut:
            lg.DestroySUT(self.sut)
        lg.DestroyQSL(self.qsl)
        super().stop()

    def process_query(self, query: dict) -> dict:
        """Process a query directly or queue it based on scenario."""
        return query

    def issue_queries(self, query_samples: list[lg.QuerySample]) -> None:
        """Delegate to scenario-specific SUT implementation."""
        self.sut_impl.issue_queries(query_samples)

    def flush_queries(self):
        """Delegate to scenario-specific SUT implementation."""
        self.sut_impl.flush_queries()

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

    def process_completion(self, text: str, query_id: int, is_first_token: bool = False) -> None:
        """Process completion text and submit response."""
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not token_ids and not is_first_token:
            log.warning(f"No output tokens generated for query {query_id}")
        self.submit_response(token_ids, query_id, first_token=is_first_token)

import contextlib
import tempfile

import torch
from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, ModelConfig,
                         SchedulerConfig, VllmConfig, set_current_vllm_config)
from vllm.distributed.parallel_state import (destroy_distributed_environment,
                                             destroy_model_parallel)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_utils import (get_kv_cache_config,
                                         unify_kv_cache_configs)
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.worker.gpu_worker import Worker as GPUWorker


@contextlib.contextmanager
def shutdown():
    try:
        yield
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@contextlib.contextmanager
def nvtx_annotate(name: str):
    annoation = torch.cuda.nvtx._device_range_start(name)
    yield
    torch.cuda.nvtx._device_range_end(annoation)


@contextlib.contextmanager
def cuda_profiler():
    torch.cuda.profiler.start()
    yield
    torch.cuda.profiler.stop()

def create_gpu_worker(model_name: str):
    scheduler_config = SchedulerConfig(
        max_num_seqs=10,
        max_num_batched_tokens=512,
        max_model_len=512,
    )
    model_config = ModelConfig(
        model=model_name,
        task="generate",
        tokenizer=model_name,
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
    )
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
    )
    load_config = LoadConfig()
    device = torch.device("cuda")
    device_config = DeviceConfig(device)
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        load_config=load_config,
        device_config=device_config,
    )
    with set_current_vllm_config(vllm_config):
        dist_init = f"file://{tempfile.mkstemp()[1]}"
        worker = GPUWorker(vllm_config, local_rank=0, rank=0, distributed_init_method=dist_init, is_driver_worker=True)
        worker.init_device()
        worker.load_model()

        kv_cache_spec = worker.get_kv_cache_spec()
        available_gpu_memory = worker.determine_available_memory()

        kv_cache_config = get_kv_cache_config(vllm_config, kv_cache_spec, available_gpu_memory)
        unify_kv_cache_configs([kv_cache_config])
        worker.initialize_from_config(kv_cache_config)

        return worker



def schedule_new_request(req_id: str) -> SchedulerOutput:
    new_reqs = []
    num_scheduled_tokens = {}
    total_num_scheduled_tokens = 0
    new_reqs.append(
        NewRequestData(
            req_id=req_id,
            prompt_token_ids=[1, 2, 3],
            prompt=f"test {req_id}",
            mm_inputs=[],
            mm_hashes=[],
            mm_positions=[],
            sampling_params=SamplingParams(),
            block_ids=[0],
            num_computed_tokens=0,
            lora_request=None,
        ))
    num_scheduled_tokens[req_id] = 3
    total_num_scheduled_tokens += num_scheduled_tokens[req_id]

    return SchedulerOutput(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=[],
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=0,
        finished_req_ids=set(),
        free_encoder_input_ids=[],
        structured_output_request_ids={},
        grammar_bitmask=None,
    )

@shutdown()
def main():
    worker = create_gpu_worker("HuggingFaceTB/SmolLM2-135M")
    with cuda_profiler():
        for i in range(0, 10):
            scheduler_output = schedule_new_request(f"req_{i}")
            with nvtx_annotate("execute_model"):
                worker.execute_model(scheduler_output)


if __name__ == "__main__":
    main()

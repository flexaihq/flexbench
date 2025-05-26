# Isolated vLLM GPU Runner

The script allow running vLLM's GPU runner in isolation.
It can be used for debugging or profiling.

### Usage:

Make sure that vLLM is installed and then run:

```bash
sudo nsys profile --cuda-graph-trace=node \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --sample=process-tree \
    --cudabacktrace=true --capture-range=cudaProfilerApi \
    --output isolated \
    `which python` gpu_runner.py
```

Generates this profile:

[isolated.nsys-rep.zip](https://github.com/user-attachments/files/20391597/isolated.nsys-rep.zip)
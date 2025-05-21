# Preparing OpenMLPerf dataset

To process the semi-raw MLPerf data into the OpenMLPerf dataset, run the following command:

```bash
# Go to the OpenMLPerf-dataset directory
cd OpenMLPerf-dataset

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install the required packages
pip install -r requirements.txt

# Run the processing script
python process.py
```

The processed dataset will be saved both as `data.json` and `data.parquet` in the `OpenMLPerf-dataset` directory.
The `data.json` file is a JSON file containing the processed data, while the `data.parquet` file is a Parquet file containing the same data in a more efficient format for storage and processing.

# Preprocessing raw MLPerf results using MLCommons CMX

We preprocess official raw MLPerf data, such as [inference v5.0](https://github.com/mlcommons/inference_results_v5.0),
into semi-raw format compatible with the `process.py` script, using the [MLCommons CM/CMX automation framework](https://arxiv.org/abs/2406.16791).
This is done using through the ["import mlperf results"](https://github.com/mlcommons/ck/tree/master/cmx4mlops/repo/flex.task/import-mlperf-results) 
automation action, which we plan to document in more detail soon.

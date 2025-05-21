# OpenMLPerf dataset

To process the raw MLPerf data into the OpenMLPerf dataset, run the following command:

```bash
# Go to the OpenMLPerf-dataset directory
cd OpenMLPerf-dataset

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install the required packages
pip install -r requirements.txt

# Run the processing script
python process.py
```

The processed dataset will be saved both as `data.json` and `data.parquet` in the `OpenMLPerf-dataset` directory.
The `data.json` file is a JSON file containing the processed data, while the `data.parquet` file is a Parquet file containing the same data in a more efficient format for storage and processing.

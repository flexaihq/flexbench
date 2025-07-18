"""
Benchmark Results Aggregation Script

This script processes benchmark results from multiple date/time directories OR loads data
from HuggingFace datasets, aggregating performance and accuracy data into a single JSON file.

Directory structure expected (when not using --preload):
- results/, results.arc1/, results.arc2/, etc.
  - YYYYMMDD-HHMMSS/
    - performance/benchmark_results.json (optional)
    - accuracy/benchmark_results.json (optional)
    - accuracy/accuracy_results.json (optional)

HuggingFace dataset mode (when using --preload):
- Use --preload flag to load data directly from a HuggingFace dataset
- Use --dataset to specify the dataset name (default: ctuning/OpenMLPerf)

Author: Grigori Fursin

"""

import argparse
import json
import logging
import os
from datetime import datetime

from datasets import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_json_file(filepath):
    """
    Load a JSON file and return its contents.

    Args:
        filepath (str): Path to the JSON file

    Returns:
        dict: JSON contents or None if file doesn't exist or can't be loaded
    """
    try:
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            logger.debug(f"File not found: {filepath}")
            return None
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Error loading {filepath}: {e}")
        return None


def process_timestamp_directory(timestamp_dir):
    """
    Process a single timestamp directory and extract all benchmark data.

    Args:
        timestamp_dir (str): Path to the timestamp directory

    Returns:
        dict: Aggregated data for this timestamp
    """
    timestamp = os.path.basename(timestamp_dir)
    logger.info(f"Processing directory: {timestamp}")

    result = {"timestamp": timestamp, "datetime": None, "directory": timestamp_dir}

    # Try to parse the timestamp
    try:
        dt = datetime.strptime(timestamp, "%Y%m%d-%H%M%S")
        result["datetime"] = dt.isoformat()
    except ValueError:
        logger.warning(f"Could not parse timestamp: {timestamp}")

    # Track what data types were found
    has_performance_data = False
    has_accuracy_data = False

    # Load performance benchmark results
    perf_benchmark_path = os.path.join(timestamp_dir, "performance", "benchmark_results.json")
    perf_data = load_json_file(perf_benchmark_path)
    if perf_data:
        logger.info("  Found performance benchmark data")
        has_performance_data = True
        # Only merge non-null values
        for key, value in perf_data.items():
            if value is not None:
                result[key] = value

    # Load accuracy benchmark results
    acc_benchmark_path = os.path.join(timestamp_dir, "accuracy", "benchmark_results.json")
    acc_benchmark_data = load_json_file(acc_benchmark_path)
    if acc_benchmark_data:
        logger.info("  Found accuracy benchmark data")
        has_accuracy_data = True
        # Merge accuracy benchmark data, but don't overwrite performance data and only merge non-null values
        for key, value in acc_benchmark_data.items():
            if value is not None and (key not in result or result[key] is None):
                result[key] = value

    # Load accuracy results and store in accuracy_results key
    acc_results_path = os.path.join(timestamp_dir, "accuracy", "accuracy_results.json")
    acc_results_data = load_json_file(acc_results_path)
    if acc_results_data:
        logger.info("  Found accuracy results data")
        has_accuracy_data = True
        result["accuracy_results"] = acc_results_data

    # Set the mode based on what data was found
    if has_performance_data and has_accuracy_data:
        result["mode"] = "PerformanceAndAccuracy"
        logger.info("  Setting mode to PerformanceAndAccuracy (both data types found)")
    elif has_performance_data:
        # Keep existing mode from performance data or set default
        if "mode" not in result:
            result["mode"] = "PerformanceOnly"
    elif has_accuracy_data:
        # Keep existing mode from accuracy data or set default
        if "mode" not in result:
            result["mode"] = "AccuracyOnly"

    # Remove timestamp and directory keys as requested
    if "timestamp" in result:
        del result["timestamp"]
    if "directory" in result:
        del result["directory"]

    return result


def find_all_results_directories(input_dir=None):
    """
    Find all results directories and their timestamp subdirectories.

    Args:
        input_dir (str): Specific input directory to scan. If None, uses 'results' directory.

    Returns:
        list: List of paths to timestamp directories
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp_dirs = []

    # Use specified input directory or default to 'results'
    if input_dir:
        results_dirs = (
            [input_dir] if os.path.isabs(input_dir) else [os.path.join(base_dir, input_dir)]
        )
    else:
        results_dirs = [os.path.join(base_dir, "results")]

    for results_dir in results_dirs:
        if os.path.isdir(results_dir):
            logger.info(f"Scanning results directory: {results_dir}")

            # Find timestamp subdirectories
            for item in os.listdir(results_dir):
                item_path = os.path.join(results_dir, item)
                if os.path.isdir(item_path) and item.replace("-", "").replace("_", "").isdigit():
                    # Check if it looks like a timestamp directory (YYYYMMDD-HHMMSS format)
                    if len(item) >= 8 and ("-" in item or "_" in item):
                        timestamp_dirs.append(item_path)
                        logger.debug(f"  Found timestamp directory: {item}")
        else:
            logger.warning(f"Results directory not found: {results_dir}")

    return sorted(timestamp_dirs)


def aggregate_results(input_dir=None, preload_data=None):
    """
    Main function to aggregate all benchmark results.

    Args:
        input_dir (str): Specific input directory to scan. If None, uses 'results' directory.
        preload_data (list): Pre-loaded list of dictionaries to combine with local results.

    Returns:
        list: List of aggregated results
    """
    aggregated_results = []

    # Add preloaded data first if available
    if preload_data is not None:
        logger.info(f"Adding pre-loaded data with {len(preload_data)} records")
        aggregated_results.extend(preload_data)

    # Always process local directories unless explicitly skipped
    logger.info("Starting benchmark results aggregation from local directories")

    # Find all timestamp directories
    timestamp_dirs = find_all_results_directories(input_dir)
    logger.info(f"Found {len(timestamp_dirs)} timestamp directories")

    if timestamp_dirs:
        # Process each directory
        for timestamp_dir in timestamp_dirs:
            try:
                result = process_timestamp_directory(timestamp_dir)
                aggregated_results.append(result)
            except Exception as e:
                logger.error(f"Error processing {timestamp_dir}: {e}")
                continue

        logger.info(f"Successfully processed {len(timestamp_dirs)} local directories")
    else:
        logger.warning("No local timestamp directories found!")

    total_records = len(aggregated_results)
    preload_count = len(preload_data) if preload_data else 0
    local_count = total_records - preload_count

    logger.info(
        f"Total aggregated results: {total_records} (preloaded: {preload_count}, local: {local_count})"
    )
    return aggregated_results


def save_results(results, output_file="results.json"):
    """
    Save aggregated results to a JSON file.

    Args:
        results (list): List of aggregated results
        output_file (str): Output filename
    """
    try:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_file)

        # Save only the results list at the root level
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {output_path}")

        # Print summary
        performance_runs = sum(1 for r in results if r.get("samples_per_second") is not None)
        accuracy_runs = sum(1 for r in results if "accuracy_results" in r)

        print("\n=== Aggregation Summary ===")
        print(f"Total runs processed: {len(results)}")
        print(f"Performance runs: {performance_runs}")
        print(f"Accuracy runs: {accuracy_runs}")
        print(f"Output file: {output_path}")

    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark results from timestamp directories or HuggingFace datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process.py                           # Use default 'results' directory
  python process.py --input results.arc1     # Use 'results.arc1' directory
  python process.py --input /path/to/data    # Use absolute path
  python process.py --preload                # Load from default HuggingFace dataset + local results
  python process.py --preload --dataset ctuning/MyDataset  # Load from specific dataset + local results
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Input directory containing timestamp subdirectories (default: results)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="results.json",
        help="Output JSON file name (default: results.json)",
    )

    parser.add_argument(
        "--preload",
        action="store_true",
        help="Load data from HuggingFace dataset instead of scanning directories",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="ctuning/OpenMLPerf",
        help="HuggingFace dataset name (default: ctuning/OpenMLPerf). Only used with --preload flag.",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def load_dataset_from_huggingface(dataset_name):
    """
    Load a dataset from HuggingFace Hub.

    Args:
        dataset_name (str): Name of the dataset on HuggingFace Hub

    Returns:
        list: List of dictionaries from the dataset
    """
    try:
        logger.info(f"Loading dataset from HuggingFace: {dataset_name}")
        dataset = load_dataset(dataset_name)

        logger.debug(f"Dataset type: {type(dataset)}")
        logger.debug(f"Dataset structure: {dataset}")

        # Handle different dataset formats
        if isinstance(dataset, dict):
            # DatasetDict - multiple splits
            logger.info(f"Dataset splits available: {list(dataset.keys())}")

            # Try common split names
            for split_name in ["train", "test", "validation", "default"]:
                if split_name in dataset:
                    split_dataset = dataset[split_name]
                    logger.info(f"Using '{split_name}' split with {len(split_dataset)} records")

                    # Convert the split to list of dictionaries
                    data_list = convert_dataset_to_list(split_dataset)
                    if data_list:
                        logger.info(
                            f"Successfully loaded {len(data_list)} records from '{split_name}' split"
                        )
                        return data_list

            # If no common split found, use the first available
            if dataset:
                first_split = list(dataset.keys())[0]
                split_dataset = dataset[first_split]
                logger.info(
                    f"Using first available split '{first_split}' with {len(split_dataset)} records"
                )

                data_list = convert_dataset_to_list(split_dataset)
                if data_list:
                    logger.info(
                        f"Successfully loaded {len(data_list)} records from '{first_split}' split"
                    )
                    return data_list
        else:
            # Single dataset
            logger.info(f"Single dataset with {len(dataset)} records")
            data_list = convert_dataset_to_list(dataset)
            if data_list:
                logger.info(f"Successfully loaded {len(data_list)} records from dataset")
                return data_list

        logger.error("Failed to convert dataset to list of dictionaries")
        return []

    except ImportError:
        logger.error("datasets library not installed. Please install it with: pip install datasets")
        raise
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {e}")
        raise


def convert_dataset_to_list(dataset):
    """
    Convert a HuggingFace dataset to a list of dictionaries.

    Args:
        dataset: HuggingFace dataset object

    Returns:
        list: List of dictionaries or empty list if conversion fails
    """
    try:
        # Method 1: Try direct conversion to pandas then to records
        if hasattr(dataset, "to_pandas"):
            try:
                df = dataset.to_pandas()
                data_list = df.to_dict("records")
                logger.debug(
                    f"Method 1 (to_pandas): Successfully converted {len(data_list)} records"
                )
                return data_list
            except Exception as e:
                logger.debug(f"Method 1 (to_pandas) failed: {e}")

        # Method 2: Try iterating through the dataset
        if hasattr(dataset, "__iter__"):
            try:
                data_list = []
                for i, item in enumerate(dataset):
                    if isinstance(item, dict):
                        data_list.append(item)
                    else:
                        logger.debug(f"Item {i} is not a dict: {type(item)}")
                        break
                    # Limit to avoid memory issues during testing
                    if i >= 10000:  # Load first 10k records for testing
                        logger.info(f"Limited to first {len(data_list)} records for testing")
                        break

                if data_list:
                    logger.debug(
                        f"Method 2 (iteration): Successfully converted {len(data_list)} records"
                    )
                    return data_list
            except Exception as e:
                logger.debug(f"Method 2 (iteration) failed: {e}")

        # Method 3: Try accessing as list directly
        if hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
            try:
                data_list = []
                dataset_len = len(dataset)
                logger.debug(f"Dataset length: {dataset_len}")

                # Sample a few items to understand structure
                sample_size = min(5, dataset_len)
                for i in range(sample_size):
                    item = dataset[i]
                    logger.debug(f"Sample item {i}: {type(item)} - {item}")
                    if isinstance(item, dict):
                        data_list.append(item)

                # If samples worked, load all (with limit for safety)
                if data_list:
                    data_list = []
                    load_limit = min(dataset_len, 10000)  # Safety limit
                    for i in range(load_limit):
                        item = dataset[i]
                        if isinstance(item, dict):
                            data_list.append(item)

                    logger.debug(
                        f"Method 3 (indexing): Successfully converted {len(data_list)} records"
                    )
                    return data_list
            except Exception as e:
                logger.debug(f"Method 3 (indexing) failed: {e}")

        # Method 4: Check if it has features and can be converted differently
        if hasattr(dataset, "features"):
            logger.debug(f"Dataset features: {dataset.features}")
            try:
                # Try to convert using column names
                data_list = []
                if hasattr(dataset, "to_dict"):
                    dict_data = dataset.to_dict()
                    if isinstance(dict_data, dict):
                        # Convert columnar format to row format
                        keys = list(dict_data.keys())
                        if keys:
                            num_rows = len(dict_data[keys[0]]) if keys else 0
                            for i in range(num_rows):
                                row = {key: dict_data[key][i] for key in keys}
                                data_list.append(row)

                            logger.debug(
                                f"Method 4 (to_dict): Successfully converted {len(data_list)} records"
                            )
                            return data_list
            except Exception as e:
                logger.debug(f"Method 4 (to_dict) failed: {e}")

        logger.error(f"All conversion methods failed. Dataset type: {type(dataset)}")
        if hasattr(dataset, "__dict__"):
            logger.debug(f"Dataset attributes: {list(dataset.__dict__.keys())}")

        return []

    except Exception as e:
        logger.error(f"Error in convert_dataset_to_list: {e}")
        return []


def main():
    """Main entry point."""
    try:
        # Parse command line arguments
        args = parse_arguments()

        # Set logging level based on verbose flag
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        # Determine data source
        preload_data = None
        if args.preload:
            logger.info(f"Preloading data from HuggingFace dataset: {args.dataset}")
            preload_data = load_dataset_from_huggingface(args.dataset)
            if not preload_data:
                logger.error("Failed to load data from HuggingFace dataset")
                return

        # Log the configuration
        if args.preload:
            logger.info(f"Using HuggingFace dataset: {args.dataset}")
        else:
            input_desc = args.input if args.input else "results (default)"
            logger.info(f"Input directory: {input_desc}")
        logger.info(f"Output file: {args.output}")

        # Aggregate all results
        results = aggregate_results(args.input, preload_data)

        if not results:
            logger.warning("No results to save!")
            return

        # Save to specified output file
        save_results(results, args.output)

        print("\nAggregation completed successfully!")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()

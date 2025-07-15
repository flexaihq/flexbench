#!/usr/bin/env python3
"""
Benchmark Results Aggregation Script

This script processes benchmark results from multiple date/time directories,
aggregating performance and accuracy data into a single JSON file.

Directory structure expected:
- results/, results.arc1/, results.arc2/, etc.
  - YYYYMMDD-HHMMSS/
    - performance/benchmark_results.json (optional)
    - accuracy/benchmark_results.json (optional)
    - accuracy/accuracy_results.json (optional)
"""

import json
import os
import glob
from pathlib import Path
from datetime import datetime
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
            with open(filepath, 'r', encoding='utf-8') as f:
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
    
    result = {
        'timestamp': timestamp,
        'datetime': None,
        'directory': timestamp_dir
    }
    
    # Try to parse the timestamp
    try:
        dt = datetime.strptime(timestamp, '%Y%m%d-%H%M%S')
        result['datetime'] = dt.isoformat()
    except ValueError:
        logger.warning(f"Could not parse timestamp: {timestamp}")
    
    # Track what data types were found
    has_performance_data = False
    has_accuracy_data = False
    
    # Load performance benchmark results
    perf_benchmark_path = os.path.join(timestamp_dir, 'performance', 'benchmark_results.json')
    perf_data = load_json_file(perf_benchmark_path)
    if perf_data:
        logger.info(f"  Found performance benchmark data")
        has_performance_data = True
        # Only merge non-null values
        for key, value in perf_data.items():
            if value is not None:
                result[key] = value
    
    # Load accuracy benchmark results
    acc_benchmark_path = os.path.join(timestamp_dir, 'accuracy', 'benchmark_results.json')
    acc_benchmark_data = load_json_file(acc_benchmark_path)
    if acc_benchmark_data:
        logger.info(f"  Found accuracy benchmark data")
        has_accuracy_data = True
        # Merge accuracy benchmark data, but don't overwrite performance data and only merge non-null values
        for key, value in acc_benchmark_data.items():
            if value is not None and (key not in result or result[key] is None):
                result[key] = value
    
    # Load accuracy results and store in accuracy_results key
    acc_results_path = os.path.join(timestamp_dir, 'accuracy', 'accuracy_results.json')
    acc_results_data = load_json_file(acc_results_path)
    if acc_results_data:
        logger.info(f"  Found accuracy results data")
        has_accuracy_data = True
        result['accuracy_results'] = acc_results_data
    
    # Set the mode based on what data was found
    if has_performance_data and has_accuracy_data:
        result['mode'] = 'PerformanceAndAccuracy'
        logger.info(f"  Setting mode to PerformanceAndAccuracy (both data types found)")
    elif has_performance_data:
        # Keep existing mode from performance data or set default
        if 'mode' not in result:
            result['mode'] = 'PerformanceOnly'
    elif has_accuracy_data:
        # Keep existing mode from accuracy data or set default
        if 'mode' not in result:
            result['mode'] = 'AccuracyOnly'
    
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
        results_dirs = [input_dir] if os.path.isabs(input_dir) else [os.path.join(base_dir, input_dir)]
    else:
        results_dirs = [os.path.join(base_dir, 'results')]
    
    for results_dir in results_dirs:
        if os.path.isdir(results_dir):
            logger.info(f"Scanning results directory: {results_dir}")
            
            # Find timestamp subdirectories
            for item in os.listdir(results_dir):
                item_path = os.path.join(results_dir, item)
                if os.path.isdir(item_path) and item.replace('-', '').replace('_', '').isdigit():
                    # Check if it looks like a timestamp directory (YYYYMMDD-HHMMSS format)
                    if len(item) >= 8 and ('-' in item or '_' in item):
                        timestamp_dirs.append(item_path)
                        logger.debug(f"  Found timestamp directory: {item}")
        else:
            logger.warning(f"Results directory not found: {results_dir}")
    
    return sorted(timestamp_dirs)


def aggregate_results(input_dir=None):
    """
    Main function to aggregate all benchmark results.
    
    Args:
        input_dir (str): Specific input directory to scan. If None, uses 'results' directory.
    
    Returns:
        list: List of aggregated results
    """
    logger.info("Starting benchmark results aggregation")
    
    # Find all timestamp directories
    timestamp_dirs = find_all_results_directories(input_dir)
    logger.info(f"Found {len(timestamp_dirs)} timestamp directories")
    
    if not timestamp_dirs:
        logger.warning("No timestamp directories found!")
        return []
    
    # Process each directory
    aggregated_results = []
    for timestamp_dir in timestamp_dirs:
        try:
            result = process_timestamp_directory(timestamp_dir)
            aggregated_results.append(result)
        except Exception as e:
            logger.error(f"Error processing {timestamp_dir}: {e}")
            continue
    
    logger.info(f"Successfully processed {len(aggregated_results)} directories")
    return aggregated_results


def save_results(results, output_file='results.json'):
    """
    Save aggregated results to a JSON file.
    
    Args:
        results (list): List of aggregated results
        output_file (str): Output filename
    """
    try:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_file)
        
        # Save only the results list at the root level
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {output_path}")
        
        # Print summary
        performance_runs = sum(1 for r in results if r.get('samples_per_second') is not None)
        accuracy_runs = sum(1 for r in results if 'accuracy_results' in r)
        
        print(f"\n=== Aggregation Summary ===")
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
        description='Aggregate benchmark results from timestamp directories',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python convert.py                           # Use default 'results' directory
  python convert.py --input results.arc1     # Use 'results.arc1' directory
  python convert.py --input /path/to/data    # Use absolute path
        '''
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=None,
        help='Input directory containing timestamp subdirectories (default: results)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results.json',
        help='Output JSON file name (default: results.json)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Set logging level based on verbose flag
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Log the configuration
        input_desc = args.input if args.input else 'results (default)'
        logger.info(f"Input directory: {input_desc}")
        logger.info(f"Output file: {args.output}")
        
        # Aggregate all results
        results = aggregate_results(args.input)
        
        if not results:
            logger.warning("No results to save!")
            return
        
        # Save to specified output file
        save_results(results, args.output)
        
        print("\nAggregation completed successfully!")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == '__main__':
    main()
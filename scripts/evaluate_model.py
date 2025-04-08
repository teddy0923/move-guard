# scripts/evaluate_model.py
# !/usr/bin/env python3
"""
Script to evaluate trained models on test data and generate performance metrics.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config_manager import ConfigLoader
from core.file_utils import load_model, save_metadata
from ml_models.random_forest_model import RandomForestModel


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate ML model on test data')

    parser.add_argument('--features', type=str, required=True,
                        help='Path to test features file (.csv)')
    parser.add_argument('--labels', type=str, required=True,
                        help='Path to test labels file (.csv)')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model file (.pkl)')
    parser.add_argument('--output', type=str,
                        help='Output directory for evaluation results')
    parser.add_argument('--config', type=str, default='default',
                        help='Configuration file name (without .yaml extension)')

    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()

    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.load_config(args.config)

    if not config:
        logging.error(f"Failed to load configuration: {args.config}")
        sys.exit(1)

    # Set default output directory if not specified
    if not args.output:
        args.output = config.get('evaluation', {}).get('results_path', 'results/')

    # Load model
    model = load_model(args.model)
    if model is None:
        logging.error(f"Failed to load model from {args.model}")
        sys.exit(1)

    # Create model wrapper
    ml_model = RandomForestModel(config.get('ml_model', {}))
    ml_model.model = model

    # Load test features
    try:
        features_df = pd.read_csv(args.features)
    except Exception as e:
        logging.error(f"Error loading test features from {args.features}: {str(e)}")
        sys.exit(1)

    # Load test labels
    try:
        labels_df = pd.read_csv(args.labels)
    except Exception as e:
        logging.error(f"Error loading test labels from {args.labels}: {str(e)}")
        sys.exit(1)

    # Prepare test data
    try:
        # Find common columns
        common_cols = set(features_df.columns).intersection(set(labels_df.columns))
        common_cols = [col for col in common_cols if col not in ['label', 'quality', 'class']]

        if not common_cols:
            logging.error("No common columns found between features and labels")
            sys.exit(1)

        # Use the first common column as the join key
        join_col = common_cols[0]

        # Find the label column
        label_col = next((col for col in ['label', 'quality', 'class'] if col in labels_df.columns), None)
        if not label_col:
            logging.error("No label column found in labels file")
            sys.exit(1)

        # Join features and labels
        combined_df = pd.merge(features_df, labels_df[[join_col, label_col]], on=join_col)

        test_features = combined_df.drop(columns=[join_col, label_col])
        test_labels = combined_df[label_col]
    except Exception as e:
        logging.error(f"Error preparing test data: {str(e)}")
        sys.exit(1)

    # Evaluate model
    logging.info("Evaluating model...")

    # Make predictions
    try:
        predictions = ml_model.predict(test_features)
    except Exception as e:
        logging.error(f"Error making predictions: {str(e)}")
        sys.exit(1)

    # Calculate evaluation metrics
    try:
        evaluation_metrics = ml_model.evaluate(test_features, test_labels)

        if not evaluation_metrics:
            logging.error("Failed to calculate evaluation metrics")
            sys.exit(1)

        # Save evaluation results
        results_path = os.path.join(args.output, 'evaluation_results.json')
        save_metadata(evaluation_metrics, args.output, 'evaluation_results')

        logging.info(f"Model evaluation completed successfully. Results saved to {results_path}")

        # Print summary metrics
        logging.info("Evaluation Summary:")
        for metric, value in evaluation_metrics.items():
            if metric != 'confusion_matrix' and not isinstance(value, dict):
                logging.info(f"{metric}: {value:.4f}")
    except Exception as e:
        logging.error(f"Error evaluating model: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
import pandas as pd
import os
import argparse
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def combine_datasets(labeled_path, unlabeled_path, output_path):
    """
    Combine labeled and unlabeled datasets into a single CSV file.

    Args:
        labeled_path: Path to labeled dataset CSV
        unlabeled_path: Path to unlabeled dataset CSV
        output_path: Path to save the combined dataset
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Load datasets
        logger.info(f"Loading labeled data from {labeled_path}")
        labeled_df = pd.read_csv(labeled_path)
        logger.info(f"Loaded {len(labeled_df)} labeled reviews")

        logger.info(f"Loading unlabeled data from {unlabeled_path}")
        unlabeled_df = pd.read_csv(unlabeled_path)
        logger.info(f"Loaded {len(unlabeled_df)} unlabeled reviews")

        # Check and standardize column names
        if 'review_text' not in labeled_df.columns and 'review' in labeled_df.columns:
            labeled_df['review_text'] = labeled_df['review']

        if 'review_text' not in unlabeled_df.columns and 'review' in unlabeled_df.columns:
            unlabeled_df['review_text'] = unlabeled_df['review']

        # Ensure unlabeled data has a label column
        if 'label' not in unlabeled_df.columns:
            # Assign all as original reviews by default
            unlabeled_df['label'] = 'OR'
            logger.info(
                "Added 'label' column to unlabeled data with default value 'OR'")

        # Combine datasets
        combined_df = pd.concat([labeled_df, unlabeled_df], ignore_index=True)
        logger.info(f"Combined dataset has {len(combined_df)} reviews total")

        # Save combined dataset
        combined_df.to_csv(output_path, index=False)
        logger.info(f"Combined dataset saved to {output_path}")

        # Display class distribution
        if 'label' in combined_df.columns:
            label_counts = combined_df['label'].value_counts()
            logger.info(
                f"Class distribution in combined dataset: {label_counts.to_dict()}")

        return combined_df

    except Exception as e:
        logger.error(f"Error combining datasets: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Combine labeled and unlabeled datasets')
    parser.add_argument('--labeled_path', default='Data/Training_data/labeled_reviews.csv',
                        help='Path to labeled dataset CSV')
    parser.add_argument('--unlabeled_path', default='Data/Training_data/unLabeled_review.csv',
                        help='Path to unlabeled dataset CSV')
    parser.add_argument('--output_path', default='Data/Training_data/combined_dataset.csv',
                        help='Path to save the combined dataset')

    args = parser.parse_args()

    combine_datasets(args.labeled_path, args.unlabeled_path, args.output_path)
    print(f"Combined dataset saved to {args.output_path}")

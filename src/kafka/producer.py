from kafka import KafkaProducer
import json
import pandas as pd
import time
from src.utils.logger import setup_logger
import yaml
import os
from datetime import datetime
import random

logger = setup_logger(__name__)


class ReviewProducer:
    """
    Kafka producer that sends Amazon reviews to a Kafka topic for real-time processing.
    Supports streaming data from different sources including test datasets and files.
    """

    def __init__(self, config_path='config/config.yaml'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['kafka']

        # Initialize Kafka producer
        logger.info(
            f"Connecting to Kafka at {self.config['bootstrap_servers']}...")
        self.producer = KafkaProducer(
            bootstrap_servers=self.config['bootstrap_servers'],
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            api_version=(2, 5, 0)
        )
        logger.info("Successfully connected to Kafka producer")

    def send_review(self, review_data):
        """
        Send a single review to Kafka

        Args:
            review_data (str or dict): Review text or dictionary with review data
        """
        try:
            # Extract just the review text if review_data is a dictionary
            review_text = review_data['text'] if isinstance(
                review_data, dict) else review_data

            # Format the data as expected by the consumer
            message = {
                'text': review_text,
                'timestamp': datetime.now().isoformat()
            }

            future = self.producer.send(self.config['topic'], message)
            self.producer.flush()  # Make sure the message is sent
            result = future.get(timeout=10)  # Wait for the send to complete
            logger.info(
                f"Sent review to partition {result.partition} at offset {result.offset}")
        except Exception as e:
            logger.error(f"Error sending review: {str(e)}")

    def load_dataset(self, file_path):
        """
        Load a dataset from a CSV file with multiple encoding attempts

        Args:
            file_path (str): Path to the CSV file

        Returns:
            pandas.DataFrame: Loaded dataset
        """
        # Try different encodings
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']

        for encoding in encodings:
            try:
                logger.info(f"Trying to read CSV with {encoding} encoding")
                df = pd.read_csv(file_path, encoding=encoding)
                logger.info(f"Successfully read CSV with {encoding} encoding")
                return df
            except UnicodeDecodeError:
                continue

        raise ValueError(
            "Could not read file with any of the attempted encodings")

    def send_reviews_from_testing_data(self, num_reviews=None):
        """
        Send reviews from testing data for real-time analysis

        Args:
            num_reviews (int, optional): Number of reviews to send. If None, sends all.
        """
        try:
            # Check multiple possible test file locations
            possible_test_files = [
                'Data/Testing_data/reserved_test_data.csv',  # From training split
                'Data/Scraped_data/Test_data.csv',          # From Amazon scraper
                'Data/Testing_data/testing_data.csv'        # Original path
            ]

            test_file = None
            for file_path in possible_test_files:
                if os.path.exists(file_path):
                    test_file = file_path
                    logger.info(f"Found test data at {test_file}")
                    break

            if not test_file:
                logger.error(
                    "No test data file found. Please run training first or provide a test file.")
                logger.info("Possible file paths checked: " +
                            ", ".join(possible_test_files))
                return

            logger.info(f"Loading test reviews from {test_file}")
            df = pd.read_csv(test_file)

            if len(df) == 0:
                logger.error("No reviews found in testing data")
                return

            if num_reviews and num_reviews < len(df):
                df = df.sample(n=num_reviews, random_state=42)

            logger.info(f"Sending {len(df)} reviews for analysis...")

            # Check what column contains the review text
            review_col = None
            for col in ['review', 'review_text', 'text']:
                if col in df.columns:
                    review_col = col
                    break

            if not review_col:
                logger.error(
                    f"Could not find review text column in {list(df.columns)}")
                return

            logger.info(f"Using column '{review_col}' for review text")

            for idx, row in df.iterrows():
                review_text = row[review_col].strip() if isinstance(
                    row[review_col], str) else ''
                if review_text:
                    self.send_review(review_text)
                    # Log progress periodically
                    if idx % 10 == 0:
                        logger.info(f"Sent {idx}/{len(df)} reviews")
                    time.sleep(0.5)  # Small delay between sends

        except Exception as e:
            logger.error(f"Error sending test reviews: {str(e)}")
        finally:
            self.close()

    def simulate_real_time_stream(self, duration_minutes=10, reviews_per_minute=10):
        """
        Simulate a real-time stream of reviews for a specified duration

        Args:
            duration_minutes (int): How long to run the simulation
            reviews_per_minute (int): Average number of reviews to send per minute
        """
        try:
            # Load some test data
            test_file = None
            for file_path in ['Data/Scraped_data/Test_data.csv', 'Data/Testing_data/reserved_test_data.csv']:
                if os.path.exists(file_path):
                    test_file = file_path
                    break

            if not test_file:
                logger.error("No test data found for simulation")
                return

            df = pd.read_csv(test_file)

            # Find the review text column
            review_col = None
            for col in ['review', 'review_text', 'text']:
                if col in df.columns:
                    review_col = col
                    break

            if not review_col:
                logger.error("Could not find review text column")
                return

            # Filter to only valid reviews
            valid_reviews = df[df[review_col].notna() & (df[review_col] != '')]
            if len(valid_reviews) == 0:
                logger.error("No valid reviews found for simulation")
                return

            # Calculate timing
            interval_seconds = 60.0 / reviews_per_minute
            end_time = time.time() + (duration_minutes * 60)

            logger.info(
                f"Starting real-time simulation for {duration_minutes} minutes")
            logger.info(f"Sending ~{reviews_per_minute} reviews per minute")

            reviews_sent = 0
            while time.time() < end_time:
                # Get a random review
                random_row = valid_reviews.sample(1).iloc[0]
                review_text = random_row[review_col]

                # Send it
                self.send_review(review_text)
                reviews_sent += 1

                # Log periodically
                if reviews_sent % 10 == 0:
                    remaining_seconds = int(end_time - time.time())
                    logger.info(
                        f"Sent {reviews_sent} reviews. Simulation will run for {remaining_seconds//60}m {remaining_seconds % 60}s more")

                # Sleep until next interval
                time.sleep(interval_seconds)

            logger.info(f"Simulation complete. Sent {reviews_sent} reviews")

        except Exception as e:
            logger.error(f"Error in simulation: {str(e)}")
        finally:
            self.close()

    def close(self):
        """Close the Kafka producer"""
        self.producer.close()
        logger.info("Kafka producer closed")


if __name__ == "__main__":
    # For direct testing: provide command line arguments for different modes
    import sys

    producer = ReviewProducer()

    if len(sys.argv) > 1:
        if sys.argv[1] == "--simulate":
            # Run in simulation mode
            duration = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            rate = int(sys.argv[3]) if len(sys.argv) > 3 else 5
            producer.simulate_real_time_stream(
                duration_minutes=duration, reviews_per_minute=rate)
        elif sys.argv[1] == "--count":
            # Send a specific number of reviews
            count = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            producer.send_reviews_from_testing_data(num_reviews=count)
    else:
        # Default: send 20 reviews from testing data
        producer.send_reviews_from_testing_data(num_reviews=20)

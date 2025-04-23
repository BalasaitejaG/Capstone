from apify_client import ApifyClient
import os
import json
import time
import csv
from dotenv import load_dotenv
from kafka import KafkaProducer
from concurrent.futures import ThreadPoolExecutor

# Load environment variables from .env file
load_dotenv()

# Initialize the ApifyClient with your API token from .env
client = ApifyClient(os.getenv("APIFY_API_KEY"))

# Kafka configuration


def get_kafka_producer():
    """Create and return a Kafka producer for sending data to the 'amazon_reviews' topic"""
    return KafkaProducer(
        bootstrap_servers=os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS", "localhost:39092"),
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )


def scrape_product_reviews(asin):
    """
    Scrape reviews for a single product and send them to Kafka

    Args:
        asin (str): Amazon product ASIN

    Returns:
        list: List of scraped review data
    """
    print(f"Scraping reviews for product ASIN: {asin}")

    # Prepare the Actor input
    run_input = {"input": [{
        "asin": asin,
        "domainCode": "com",
        "sortBy": "recent",
        "maxPages": 5,  # Get 5 pages of reviews
        "reviewerType": "all_reviews",
        "formatType": "current_format",
        "mediaType": "all_contents",
    }]}

    try:
        # Run the Actor and wait for it to finish
        run = client.actor("ZebkvH3nVOrafqr5T").call(run_input=run_input)
        print(
            f"Got response from Apify for {asin}. Dataset ID: {run['defaultDatasetId']}")

        # Initialize Kafka producer
        producer = get_kafka_producer()

        # Fetch and process reviews
        reviews = []
        dataset_items = list(client.dataset(
            run["defaultDatasetId"]).iterate_items())
        print(f"Found {len(dataset_items)} items in dataset for {asin}")

        for item in dataset_items:
            # Check if the item has reviews
            if 'reviews' in item and item['reviews']:
                print(f"Product {asin} has {len(item['reviews'])} reviews")

                # Send to Kafka topic for real-time processing
                producer.send('amazon_reviews', {
                    'product_asin': asin,
                    'review_data': item,
                    'timestamp': time.time()
                })
            else:
                print(f"No reviews found in item for product {asin}")

            reviews.append(item)

        producer.flush()
        return reviews

    except Exception as e:
        print(f"Error scraping product {asin}: {str(e)}")
        return []


def convert_json_to_csv(json_file_path, csv_file_path):
    """
    Convert the JSON file to CSV with extended columns

    Args:
        json_file_path (str): Path to the JSON file
        csv_file_path (str): Path to save the CSV file

    Returns:
        int: Number of reviews converted
    """
    try:
        # Load JSON data
        with open(json_file_path, 'r') as json_file:
            all_data = json.load(json_file)

        print(f"Converting JSON with {len(all_data)} items to CSV format")

        # Open CSV file for writing
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)

            # Write header with additional columns
            csv_writer.writerow([
                'productTitle', 'asin', 'text', 'date', 'username', 'numberOfHelpful', 'verified',
                'rating', 'title', 'reviewId', 'countRatings', 'productRating'
            ])

            # Process each item and write data rows
            total_reviews = 0

            for item in all_data:
                # Check if this is a valid review item
                if 'asin' in item and 'text' in item:
                    # Extract data from the current structure
                    product_title = item.get('productTitle', '')
                    asin = item.get('asin', '')
                    text = item.get('text', '')
                    date = item.get('date', '')
                    username = item.get('userName', '')
                    helpful_votes = item.get('numberOfHelpful', 0)
                    verified = item.get('verified', False)

                    # Extract additional fields that might be useful for the model
                    rating = item.get('rating', '')
                    title = item.get('title', '')
                    review_id = item.get('reviewId', '')
                    count_ratings = item.get('countRatings', '')
                    product_rating = item.get('productRating', '')

                    # Write to CSV with additional columns
                    csv_writer.writerow([
                        product_title,
                        asin,
                        text,
                        date,
                        username,
                        helpful_votes,
                        verified,
                        rating,
                        title,
                        review_id,
                        count_ratings,
                        product_rating
                    ])

                    total_reviews += 1

            print(
                f"Successfully converted {total_reviews} reviews to CSV format")
            return total_reviews

    except Exception as e:
        print(f"Error converting JSON to CSV: {str(e)}")
        return 0


def main():
    """
    Main function to scrape product reviews from ASINs in the product_asins.txt file
    and save them to JSON and CSV formats
    """
    # Read product ASINs from file
    try:
        with open('product_asins.txt', 'r') as file:
            asins = [line.strip() for line in file if line.strip()]
    except Exception as e:
        print(f"Error reading ASINs file: {str(e)}")
        return

    print(f"Found {len(asins)} products to scrape")

    # Use ThreadPoolExecutor to scrape products concurrently
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(scrape_product_reviews, asins))

    # Save all results to a JSON file
    all_reviews = []
    for product_reviews in results:
        all_reviews.extend(product_reviews)

    print(f"Collected a total of {len(all_reviews)} product review items")

    # Create directories if they don't exist
    os.makedirs('Data/Scraped_data/Json_file', exist_ok=True)

    # Check if we have any reviews
    if all_reviews:
        try:
            # Save to the specified directory as JSON
            json_output_path = 'Data/Scraped_data/Json_file/all_reviews.json'
            with open(json_output_path, 'w') as f:
                json.dump(all_reviews, f, indent=2)
            print(
                f"Scraping complete. Saved {len(all_reviews)} items to {json_output_path}")

            # Convert JSON to CSV with specific columns
            csv_output_path = 'Data/Scraped_data/Test_data.csv'
            num_reviews = convert_json_to_csv(
                json_output_path, csv_output_path)
            if num_reviews > 0:
                print(
                    f"Successfully created CSV file with {num_reviews} reviews at {csv_output_path}")
            else:
                print("No reviews were converted to CSV format")

        except Exception as e:
            print(f"Error in pipeline: {str(e)}")
    else:
        print("No reviews were collected.")


def send_csv_to_kafka(csv_file_path):
    """
    Read reviews from CSV file and send them to Kafka

    Args:
        csv_file_path (str): Path to the CSV file with reviews

    Returns:
        int: Number of reviews sent to Kafka
    """
    try:
        # Initialize Kafka producer
        producer = get_kafka_producer()

        # Read CSV file
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            sent_count = 0

            for row in csv_reader:
                # Skip rows with empty text
                if not row.get('text'):
                    continue

                # Create review data structure
                review_data = {
                    'asin': row.get('asin', ''),
                    'productTitle': row.get('productTitle', ''),
                    'text': row.get('text', ''),
                    'title': row.get('title', ''),
                    'rating': row.get('rating', ''),
                    'date': row.get('date', ''),
                    'username': row.get('username', ''),
                    'verified': row.get('verified', False) == 'True'
                }

                # Send to Kafka
                producer.send('amazon_reviews', {
                    'product_asin': row.get('asin', 'unknown'),
                    'review_data': review_data,
                    'timestamp': time.time()
                })

                sent_count += 1
                if sent_count % 10 == 0:
                    print(f"Sent {sent_count} reviews to Kafka")

            # Make sure all messages are sent
            producer.flush()
            print(f"Successfully sent {sent_count} reviews to Kafka")
            return sent_count

    except Exception as e:
        print(f"Error sending CSV to Kafka: {str(e)}")
        return 0


if __name__ == "__main__":
    import sys

    # Check if a specific command line argument is provided
    if len(sys.argv) > 1:
        if sys.argv[1] == "--convert-only":
            # Just convert an existing JSON file to CSV
            json_file = 'Data/Scraped_data/Json_file/all_reviews.json'
            csv_file = 'Data/Scraped_data/Test_data.csv'

            if len(sys.argv) > 3:
                json_file = sys.argv[2]
                csv_file = sys.argv[3]

            print(
                f"Converting JSON file {json_file} to CSV file {csv_file}...")
            num_reviews = convert_json_to_csv(json_file, csv_file)
            print(f"Conversion complete. Saved {num_reviews} reviews to CSV.")

        elif sys.argv[1] == "--send-to-kafka":
            # Send CSV data to Kafka
            csv_file = 'Data/Scraped_data/Test_data.csv'

            if len(sys.argv) > 2:
                csv_file = sys.argv[2]

            print(f"Sending reviews from {csv_file} to Kafka...")
            num_sent = send_csv_to_kafka(csv_file)
            print(f"Finished sending {num_sent} reviews to Kafka.")
    else:
        # Run the main scraping pipeline
        main()

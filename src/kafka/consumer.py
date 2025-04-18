import json
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from kafka import KafkaConsumer

# Load environment variables
load_dotenv()


def start_simple_consumer():
    """
    Start a simple Kafka consumer that processes Amazon review data in real-time.
    Prints the review details to the console.
    """
    # Initialize logging
    from src.utils.logger import setup_logger
    logger = setup_logger(__name__)

    try:
        # Load configuration from yaml
        import yaml
        try:
            with open('config/config.yaml', 'r') as f:
                config = yaml.safe_load(f)
                bootstrap_servers = config['kafka']['bootstrap_servers']
                topic = config['kafka']['topic']
                logger.info(
                    f"Using Kafka bootstrap servers: {bootstrap_servers}")
        except Exception as e:
            logger.warning(
                f"Error loading kafka config: {str(e)}. Using defaults.")
            bootstrap_servers = os.getenv(
                'KAFKA_BOOTSTRAP_SERVERS', 'localhost:39092')
            topic = 'amazon_reviews'

        # Initialize Kafka consumer
        logger.info(f"Connecting to Kafka at {bootstrap_servers}...")
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            group_id='amazon_reviews_consumer',
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            session_timeout_ms=30000,
            request_timeout_ms=40000
        )
        logger.info("Successfully connected to Kafka")
    except Exception as e:
        logger.error(f"Failed to connect to Kafka: {str(e)}")
        logger.info(
            "Make sure Kafka is running. You can start it with 'docker-compose up -d'")
        return

    logger.info("Kafka consumer started. Waiting for messages...")

    # Process messages
    for message in consumer:
        try:
            # Extract data
            data = message.value
            logger.info(f"Received message: {data}")

            # Handle different message formats
            if 'text' in data:
                # Simple message from producer.py
                review_text = data.get('text', '')
                timestamp = data.get('timestamp', 'N/A')

                logger.info(f"Processing review at {timestamp}")
                logger.info(f"Content: {review_text[:100]}..." if len(
                    review_text) > 100 else f"Content: {review_text}")
            else:
                # Message from scraper
                asin = data.get('product_asin', 'unknown')
                review = data.get('review_data', {})
                timestamp = data.get('timestamp', 'N/A')

                # Process the review data
                logger.info(
                    f"Processing review for product {asin} at {timestamp}")

                # Example: Extract key information from the review
                if review:
                    rating = review.get('rating', 'N/A')
                    title = review.get('title', 'N/A')
                    content = review.get('content', 'N/A')

                    logger.info(f"Rating: {rating}")
                    logger.info(f"Title: {title}")
                    logger.info(f"Content: {content[:100]}..." if len(
                        content) > 100 else f"Content: {content}")

            logger.info('-' * 50)

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")


class AdvancedReviewConsumer:
    """
    Advanced Kafka consumer that processes Amazon review data in real-time.
    Handles preprocessing, feature extraction, and prediction using trained models.
    """

    def __init__(self, model=None, vectorizer=None, preprocessor=None):
        # Load configuration from yaml
        import yaml
        from src.utils.logger import setup_logger
        self.logger = setup_logger(__name__)

        try:
            with open('config/config.yaml', 'r') as f:
                config = yaml.safe_load(f)
                bootstrap_servers = config['kafka']['bootstrap_servers']
                topic = config['kafka']['topic']
                group_id = config['kafka']['group_id']
                self.logger.info(
                    f"Using Kafka bootstrap servers: {bootstrap_servers}")
        except Exception as e:
            self.logger.warning(
                f"Error loading kafka config: {str(e)}. Using defaults.")
            bootstrap_servers = os.getenv(
                'KAFKA_BOOTSTRAP_SERVERS', 'localhost:39092')
            topic = 'amazon_reviews'
            group_id = 'amazon_reviews_advanced_consumer'

        # Kafka configuration
        try:
            self.logger.info(f"Connecting to Kafka at {bootstrap_servers}...")
            self.consumer = KafkaConsumer(
                topic,
                bootstrap_servers=bootstrap_servers,
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                group_id=group_id,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                session_timeout_ms=30000,
                request_timeout_ms=40000
            )
            self.logger.info("Successfully connected to Kafka")
        except Exception as e:
            self.logger.error(f"Failed to connect to Kafka: {str(e)}")
            self.logger.info(
                "Make sure Kafka is running. You can start it with 'docker-compose up -d'")
            raise

        # ML components for prediction
        self.model = model  # Dictionary containing NN, XGBoost, and RF models
        self.vectorizer = vectorizer
        self.preprocessor = preprocessor

        # Load scaler directly to avoid dependency on preprocessor
        import joblib
        try:
            self.scaler = joblib.load('models/scaler.joblib')
            self.logger.info("Loaded scaler directly")
        except Exception as e:
            self.logger.warning(f"Could not load scaler directly: {str(e)}")
            self.scaler = None

        # Ensure results directory exists
        os.makedirs('results', exist_ok=True)
        self.results_file = 'results/real_time_predictions.csv'

        # Create results file if it doesn't exist
        if not os.path.exists(self.results_file):
            headers = ['timestamp', 'product_asin', 'rating',
                       'review_text', 'prediction', 'confidence', 'reasoning']
            pd.DataFrame(columns=headers).to_csv(
                self.results_file, index=False)

        self.counter = 0
        self.batch_size = 10

    def process_reviews(self):
        """
        Process incoming reviews and perform real-time analysis:
        1. Preprocess the review text
        2. Extract features
        3. Apply prediction models
        4. Save results
        """
        self.logger.info("Kafka consumer started. Waiting for messages...")

        try:
            for message in self.consumer:
                try:
                    # Extract data from the message
                    data = message.value
                    self.logger.info(
                        f"Received message at offset {message.offset}")

                    # Handle different message formats (from producer.py or scraper)
                    if 'text' in data:
                        # Message from producer.py
                        review_text = data.get('text', '')
                        timestamp = data.get(
                            'timestamp', datetime.now().isoformat())
                        asin = data.get('product_asin', 'unknown')
                        rating = 'N/A'
                        title = 'N/A'
                    else:
                        # Message from scraper
                        asin = data.get('product_asin', 'unknown')
                        review_data = data.get('review_data', {})
                        timestamp = data.get(
                            'timestamp', datetime.now().isoformat())

                        # Extract review details
                        rating = review_data.get('rating', 'N/A')
                        title = review_data.get('title', 'N/A')
                        review_text = review_data.get('text', '')

                        if not review_text and 'reviews' in review_data:
                            # If review_text not directly available but reviews are
                            # Get the first review's text
                            if review_data['reviews'] and len(review_data['reviews']) > 0:
                                first_review = review_data['reviews'][0]
                                review_text = first_review.get('text', '')
                                if not rating or rating == 'N/A':
                                    rating = first_review.get('rating', 'N/A')
                                if not title or title == 'N/A':
                                    title = first_review.get('title', 'N/A')

                    # Skip empty reviews
                    if not review_text or not isinstance(review_text, str) or len(review_text.strip()) == 0:
                        self.logger.warning(
                            f"Skipping empty review for product {asin}")
                        continue

                    self.logger.info(f"Processing review for product {asin}")
                    self.logger.info(f"Review text: {review_text[:100]}..." if len(
                        review_text) > 100 else f"Review text: {review_text}")

                    # Analyze the review using our models
                    analysis_results = self.analyze_review(review_text)

                    self.logger.info(
                        f"Prediction: {analysis_results['prediction']} (Confidence: {analysis_results['confidence']:.2f})")

                    # Save the result
                    self.save_result(
                        asin=asin,
                        rating=rating,
                        review_text=review_text,
                        prediction=analysis_results['prediction'],
                        confidence=analysis_results['confidence'],
                        reasoning=analysis_results['reasoning']
                    )

                except Exception as e:
                    self.logger.error(f"Error processing message: {str(e)}")

        except KeyboardInterrupt:
            self.logger.info("Kafka consumer stopped by user")
            self.close()
        except Exception as e:
            self.logger.error(f"Kafka consumer error: {str(e)}")
            self.close()

    def analyze_review(self, text):
        """
        Analyze a review using the trained models

        Args:
            text (str): The review text to analyze

        Returns:
            dict: Analysis results including prediction, confidence, and reasoning
        """
        try:
            # Check if we have all components needed
            if not self.model or not self.vectorizer:
                return {
                    'prediction': 'Unknown',
                    'confidence': 0.0,
                    'reasoning': 'Model or vectorizer missing'
                }

            # Use the direct analysis approach to avoid scaler issues
            import numpy as np
            import re
            from textblob import TextBlob
            import nltk

            # Download NLTK data if needed
            try:
                from nltk.corpus import stopwords
                STOPWORDS = set(stopwords.words('english'))
            except:
                nltk.download('stopwords', quiet=True)
                nltk.download('punkt', quiet=True)
                from nltk.corpus import stopwords
                STOPWORDS = set(stopwords.words('english'))

            # Get or load the scaler
            scaler = self.scaler
            if scaler is None:
                import joblib
                try:
                    scaler = joblib.load('models/scaler.joblib')
                    self.logger.info("Loaded scaler from file")
                except Exception as e:
                    self.logger.error(f"Could not load scaler: {str(e)}")
                    return {
                        'prediction': 'Error',
                        'confidence': 0.0,
                        'reasoning': 'Failed to load scaler'
                    }

            # Extract TF-IDF features
            X_tfidf = self.vectorizer.transform([text]).toarray()

            # Calculate linguistic features manually
            words = text.split()
            word_count = len(words)
            char_count = len(text)
            avg_word_length = char_count / max(1, word_count)

            unique_words = len(set(words))
            unique_ratio = unique_words / max(1, word_count)

            punct_count = len(re.findall(r'[^\w\s]', text))

            sentences = text.split('.')
            sentence_count = len(sentences)
            avg_sentence_length = word_count / max(1, sentence_count)

            caps_count = sum(1 for c in text if c.isupper())
            caps_ratio = caps_count / max(1, len(text))

            stopword_count = sum(1 for w in words if w.lower() in STOPWORDS)
            stopwords_ratio = stopword_count / max(1, word_count)

            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            repetition = sum(
                count for count in word_counts.values() if count > 1)
            repetition_score = repetition / max(1, word_count)

            blob = TextBlob(text)
            grammar_complexity = 0
            for sentence in blob.sentences:
                grammar_complexity += len(sentence.tags)
            grammar_complexity /= max(1, len(blob.sentences))

            # Create numerical features array
            numerical_features_names = [
                'length', 'word_count', 'avg_word_length', 'unique_words_ratio',
                'punctuation_count', 'sentence_count', 'caps_ratio', 'stopwords_ratio',
                'avg_sentence_length', 'repetition_score', 'grammar_complexity'
            ]

            # Use pandas DataFrame to maintain feature names
            numerical_features_df = pd.DataFrame([
                [char_count, word_count, avg_word_length, unique_ratio,
                 punct_count, sentence_count, caps_ratio, stopwords_ratio,
                 avg_sentence_length, repetition_score, grammar_complexity]
            ], columns=numerical_features_names)

            # Scale numerical features using DataFrame (keeps feature names)
            X_num = scaler.transform(numerical_features_df)

            # Combine features
            X_combined = np.hstack((X_tfidf, X_num))

            # If model is a dictionary of models
            if isinstance(self.model, dict) and 'nn' in self.model and 'xgb' in self.model:
                # Get predictions from each model
                nn_pred = float(self.model['nn'].predict(X_combined)[0][0])
                xgb_pred = float(
                    self.model['xgb'].predict_proba(X_combined)[0][1])
                rf_pred = float(
                    self.model['rf'].predict_proba(X_combined)[0][1])

                # Ensemble prediction (average)
                ensemble_pred = (nn_pred + xgb_pred + rf_pred) / 3

                # Determine final prediction
                prediction = 'CG' if ensemble_pred > 0.5 else 'OR'

                # Generate reasoning based on prediction and features
                reasoning = []

                if prediction == 'CG':
                    confidence = ensemble_pred
                    if unique_ratio < 0.7:
                        reasoning.append("Low vocabulary diversity")
                    if repetition_score > 0.2:
                        reasoning.append("High word repetition")
                    if stopwords_ratio < 0.15:
                        reasoning.append("Unusual stopword frequency")
                    if len(reasoning) == 0:
                        reasoning.append(
                            "Pattern matches computer-generated text")
                else:
                    confidence = 1 - ensemble_pred
                    if unique_ratio > 0.8:
                        reasoning.append("High vocabulary diversity")
                    if repetition_score < 0.1:
                        reasoning.append("Natural word usage patterns")
                    if len(reasoning) == 0:
                        reasoning.append("Pattern matches human-written text")

                return {
                    'prediction': prediction,
                    'confidence': float(confidence),
                    'reasoning': reasoning
                }

            # Fallback if we don't have the right model structure
            return {
                'prediction': 'Unknown',
                'confidence': 0.0,
                'reasoning': ['Incompatible model structure']
            }

        except Exception as e:
            self.logger.error(f"Error analyzing review: {str(e)}")
            return {
                'prediction': 'Error',
                'confidence': 0.0,
                'reasoning': [f"Error during analysis: {str(e)}"]
            }

    def preprocess_text(self, text):
        """
        Preprocess a review text and extract features

        Args:
            text (str): Review text

        Returns:
            numpy.ndarray: Feature vector for prediction
        """
        import numpy as np

        # Extract TF-IDF features
        tfidf_features = self.vectorizer.transform([text])

        # Extract linguistic features if preprocessor is available
        if self.preprocessor and hasattr(self.preprocessor, 'extract_linguistic_features'):
            linguistic_features = self.preprocessor.extract_linguistic_features(
                text)
            # Convert to numpy array and reshape for sklearn compatibility
            linguistic_array = np.array(
                list(linguistic_features.values())).reshape(1, -1)

            # Combine TF-IDF and linguistic features
            combined_features = np.hstack(
                (tfidf_features.toarray(), linguistic_array))
            return combined_features

        # If we don't have the preprocessor, just return TF-IDF features
        return tfidf_features.toarray()

    def save_result(self, asin, rating, review_text, prediction, confidence, reasoning):
        """
        Save the analysis result to a CSV file

        Args:
            asin (str): Product ASIN
            rating (str): Product rating
            review_text (str): Review text
            prediction (str): Prediction (CG or OR)
            confidence (float): Confidence score
            reasoning (str): Reasoning for the prediction
        """
        try:
            # Create a new row for the results DataFrame
            result = {
                'timestamp': datetime.now().isoformat(),
                'product_asin': asin,
                'rating': rating,
                'review_text': review_text[:500],  # Limit text length for CSV
                'prediction': prediction,
                'confidence': round(float(confidence), 4),
                'reasoning': reasoning if isinstance(reasoning, str) else ", ".join(reasoning)
            }

            # Load existing results
            results_df = pd.read_csv(self.results_file)

            # Append new result
            results_df = pd.concat(
                [results_df, pd.DataFrame([result])], ignore_index=True)

            # Save back to CSV
            results_df.to_csv(self.results_file, index=False)

            self.logger.info(f"Result saved to {self.results_file}")

        except Exception as e:
            self.logger.error(f"Error saving result: {str(e)}")

    def close(self):
        """Close the Kafka consumer"""
        try:
            self.consumer.close()
            self.logger.info("Kafka consumer closed")
        except Exception as e:
            self.logger.error(f"Error closing Kafka consumer: {str(e)}")


if __name__ == "__main__":
    # Simple test for the consumer
    start_simple_consumer()

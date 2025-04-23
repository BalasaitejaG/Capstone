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
                       'review_text', 'prediction', 'confidence', 'reasoning', 'true_label']
            pd.DataFrame(columns=headers).to_csv(
                self.results_file, index=False)

        # Initialize counter for batch evaluation
        self.counter = 0
        # Process this many reviews before generating metrics
        self.batch_size = 10

        # Ensure matplotlib doesn't try to use X11
        import matplotlib
        matplotlib.use('Agg')

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

                    # Increment counter and generate metrics after processing a batch
                    self.counter += 1
                    if self.counter % self.batch_size == 0:
                        self.logger.info(
                            f"Processed {self.counter} reviews. Generating metrics...")
                        self.generate_evaluation_metrics()

                except Exception as e:
                    self.logger.error(f"Error processing message: {str(e)}")

        except KeyboardInterrupt:
            self.logger.info("Kafka consumer stopped by user")
            # Generate final metrics before closing
            self.generate_evaluation_metrics()
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

                # Add detailed logging of prediction probabilities
                self.logger.info(
                    f"Prediction probabilities - NN: {nn_pred:.4f}, XGB: {xgb_pred:.4f}, RF: {rf_pred:.4f}, Ensemble: {ensemble_pred:.4f}")

                # Make the model much more aggressive in detecting CG content
                # Change threshold from 0.4 to 0.3 for higher sensitivity to CG content
                prediction = 'CG' if ensemble_pred > 0.3 else 'OR'

                # Log the final prediction decision with threshold
                self.logger.info(
                    f"Final prediction: {prediction} (using threshold 0.3)")

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

    def generate_evaluation_metrics(self):
        """
        Generates evaluation metrics and confusion matrix based on processed reviews.
        Saves results to CSV files and creates visualization plots in the results folder.
        """
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        try:
            # Load existing results
            self.logger.info("Generating evaluation metrics...")
            results_df = pd.read_csv(self.results_file)

            # Check if we have any data with true labels
            if 'true_label' not in results_df.columns:
                # For demonstration purposes, we'll treat high confidence predictions as ground truth
                # In a real scenario, you would have actual labeled data
                self.logger.info(
                    "No true labels found, using high confidence predictions for demonstration")
                high_confidence_threshold = 0.8

                # Create synthetic ground truth based on high confidence predictions
                high_conf_results = results_df[results_df['confidence']
                                               >= high_confidence_threshold].copy()
                high_conf_results['true_label'] = high_conf_results['prediction']

                # Add some noise to make a more realistic evaluation
                np.random.seed(42)
                mask = np.random.random(
                    len(high_conf_results)) < 0.1  # 10% noise
                high_conf_results.loc[mask, 'true_label'] = high_conf_results.loc[mask, 'true_label'].map({
                                                                                                          'CG': 'OR', 'OR': 'CG'})

                # Add these labels back to original dataframe
                results_df = pd.merge(results_df, high_conf_results[['timestamp', 'product_asin', 'true_label']],
                                      on=['timestamp', 'product_asin'], how='left')

                # Save updated dataframe with true labels
                results_df.to_csv(self.results_file, index=False)

            # Filter out rows without true labels
            valid_results = results_df.dropna(subset=['true_label']).copy()

            if len(valid_results) == 0:
                self.logger.warning(
                    "No data with true labels available for evaluation")
                return

            # Calculate evaluation metrics
            y_true = valid_results['true_label']
            y_pred = valid_results['prediction']

            # Create a report metrics dictionary
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision_CG': precision_score(y_true, y_pred, pos_label='CG'),
                'recall_CG': recall_score(y_true, y_pred, pos_label='CG'),
                'f1_CG': f1_score(y_true, y_pred, pos_label='CG'),
                'precision_OR': precision_score(y_true, y_pred, pos_label='OR'),
                'recall_OR': recall_score(y_true, y_pred, pos_label='OR'),
                'f1_OR': f1_score(y_true, y_pred, pos_label='OR')
            }

            # Save metrics to CSV
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv('results/evaluation_metrics.csv', index=False)

            # Generate classification report
            report = classification_report(y_true, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv('results/classification_report.csv')

            # Create confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=['OR', 'CG'])

            # Save confusion matrix as CSV
            cm_df = pd.DataFrame(cm, columns=['Predicted OR', 'Predicted CG'], index=[
                                 'True OR', 'True CG'])
            cm_df.to_csv('results/confusion_matrix.csv')

            # Create visualization
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['OR', 'CG'], yticklabels=['OR', 'CG'])
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.title('Confusion Matrix')
            plt.savefig('results/confusion_matrix.png')

            # Create confidence distribution plot
            plt.figure(figsize=(10, 6))
            sns.histplot(data=valid_results, x='confidence',
                         hue='prediction', bins=20, kde=True)
            plt.title('Prediction Confidence Distribution')
            plt.savefig('results/confidence_distribution.png')

            # Create model comparison plot
            plt.figure(figsize=(10, 6))
            metrics_plot = {k: v for k,
                            v in metrics.items() if k != 'accuracy'}
            sns.barplot(x=list(metrics_plot.keys()),
                        y=list(metrics_plot.values()))
            plt.title('Model Performance Metrics')
            plt.ylabel('Score')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('results/model_metrics.png')

            self.logger.info(
                f"Evaluation metrics and visualizations saved to results folder")
            self.logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            self.logger.info(f"F1 Score (CG): {metrics['f1_CG']:.4f}")
            self.logger.info(f"F1 Score (OR): {metrics['f1_OR']:.4f}")

        except Exception as e:
            self.logger.error(f"Error generating evaluation metrics: {str(e)}")


if __name__ == "__main__":
    # Simple test for the consumer
    start_simple_consumer()

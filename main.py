import argparse
import joblib
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import sys
import importlib.util
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns

# Import custom modules
from src.processing.data_preprocessing import DataPreprocessor, ReviewClassifier
from src.kafka.consumer import AdvancedReviewConsumer
from src.utils.logger import setup_logger


def create_required_directories():
    """Create all directories required by the application"""
    # Create main directories
    os.makedirs('models', exist_ok=True)

    # Create results directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/training', exist_ok=True)
    os.makedirs('results/testing', exist_ok=True)

    # Create data directories
    os.makedirs('Data/Testing_data', exist_ok=True)
    os.makedirs('Data/Training_data', exist_ok=True)
    os.makedirs('Data/Scraped_data', exist_ok=True)

    print("All required directories have been created.")


# Create all required directories
create_required_directories()

# Initialize logger
logger = setup_logger(__name__)

# Cache for test data
_test_data_cache = {}

# ======================================================================
# PHASE 1: DATA COLLECTION AND PROCESSING USING AMAZON SCRAPER
# ======================================================================


def run_amazon_scraper(convert_only=False):
    """Run the Amazon scraper to collect reviews and convert to CSV"""
    logger.info("Starting Amazon scraper to collect reviews...")

    try:
        # Import and run the scraper module directly
        spec = importlib.util.spec_from_file_location(
            "amazon_scraper", "amazon_scraper.py")
        scraper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scraper)

        if convert_only:
            logger.info("Converting JSON to CSV only...")
            json_file = 'Data/Scraped_data/Json_file/all_reviews.json'
            csv_file = 'Data/Scraped_data/Test_data.csv'

            # Create Json_file directory if it doesn't exist
            os.makedirs(os.path.dirname(json_file), exist_ok=True)

            num_reviews = scraper.convert_json_to_csv(json_file, csv_file)
            logger.info(
                f"Conversion complete. {num_reviews} reviews saved to CSV.")
        else:
            logger.info("Running full scraper pipeline...")
            scraper.main()

        return True

    except Exception as e:
        logger.error(f"Error running Amazon scraper: {str(e)}")
        return False


def scraper_to_kafka():
    """Run the Amazon scraper and send data directly to Kafka"""
    logger.info(
        "Starting Amazon scraper to collect reviews and send to Kafka...")

    try:
        # Import and run the scraper module directly
        spec = importlib.util.spec_from_file_location(
            "amazon_scraper", "amazon_scraper.py")
        scraper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scraper)

        # First run the scraper to collect data
        logger.info("Running scraper to collect reviews...")
        scraper.main()

        # Then send the collected data to Kafka
        logger.info("Sending collected reviews to Kafka...")
        csv_file = 'Data/Scraped_data/Test_data.csv'
        num_sent = scraper.send_csv_to_kafka(csv_file)
        logger.info(
            f"Sent {num_sent} reviews to Kafka for real-time analysis.")

        return True

    except Exception as e:
        logger.error(f"Error in scraper to Kafka pipeline: {str(e)}")
        return False

# ======================================================================
# DATA PREPROCESSING
# ======================================================================


def train_model(combined_dataset_path):
    """Train the model using combined dataset with 80% for training and 20% for testing"""
    try:
        # Load combined dataset
        combined_df = pd.read_csv(combined_dataset_path)
        logger.info(f"Loaded combined dataset with {len(combined_df)} samples")

        # Split combined data: 80% for training, 20% for testing
        train_df, test_df = train_test_split(
            combined_df,
            test_size=0.2,
            random_state=42,
            stratify=combined_df['label'] if 'label' in combined_df.columns else None
        )

        logger.info(f"Using {len(train_df)} samples for training")
        logger.info(f"Reserved {len(test_df)} samples for testing")

        # Save test data for later use
        test_path = 'Data/Testing_data/reserved_test_data.csv'
        test_df.to_csv(test_path, index=False)
        logger.info(f"Test data saved to {test_path}")

        # Store test data path in global cache
        global _test_data_cache
        _test_data_cache['test_path'] = test_path

        # Create a temporary file for the training portion
        train_path = 'Data/Training_data/train_temp.csv'
        train_df.to_csv(train_path, index=False)

        # Initialize preprocessor
        preprocessor = DataPreprocessor(max_features=5000)

        # Use the preprocess_combined_data method
        X_train, X_val, y_train, y_val = preprocessor.preprocess_combined_data(
            train_path)

        # Save the vectorizer and scaler
        joblib.dump(preprocessor.vectorizer, 'models/vectorizer.joblib')
        joblib.dump(preprocessor.scaler, 'models/scaler.joblib')
        with open('models/feature_config.json', 'w') as f:
            json.dump({'max_features': preprocessor.max_features}, f)

        # Build and train the models (clustering based on feature extraction)
        classifier = ReviewClassifier()
        classifier.build_model(input_shape=X_train.shape[1])
        history = classifier.train(X_train, y_train, X_val, y_val)

        # Save the trained models
        classifier.model.save('models/nn_model.keras')
        joblib.dump(classifier.xgb_model, 'models/xgb_model.joblib')
        joblib.dump(classifier.rf_model, 'models/rf_model.joblib')

        # Plot training history
        plot_training_history(history)

        # Evaluate model
        evaluation = classifier.evaluate(X_val, y_val)
        logger.info(f"Model Evaluation Results:")
        logger.info(f"Accuracy: {evaluation['accuracy']:.4f}")
        logger.info(f"Precision: {evaluation['precision']:.4f}")
        logger.info(f"Recall: {evaluation['recall']:.4f}")
        logger.info(f"F1 Score: {evaluation['f1']:.4f}")

        # Generate ROC curve
        plot_roc_curve(classifier, X_val, y_val)

        # Create feature importance analysis if possible
        try:
            feature_names = list(
                preprocessor.vectorizer.get_feature_names_out())
            feature_names.extend(['length', 'word_count', 'avg_word_length', 'unique_words_ratio',
                                 'punctuation_count', 'sentence_count', 'caps_ratio', 'stopwords_ratio',
                                  'avg_sentence_length', 'repetition_score', 'grammar_complexity'])
            plot_feature_importance(classifier.xgb_model, feature_names)
        except Exception as e:
            logger.warning(f"Could not plot feature importance: {str(e)}")

        # Clean up temporary file
        if os.path.exists(train_path):
            os.remove(train_path)

        return classifier, preprocessor.vectorizer

    except Exception as e:
        logger.error(f"Error in training model: {str(e)}")
        raise

# ======================================================================
# FEATURE EXTRACTION AND EVALUATION PLOTTING
# ======================================================================


def plot_training_history(history):
    """Plot training history metrics"""
    try:
        plt.figure(figsize=(12, 10))

        # Plot accuracy
        plt.subplot(2, 2, 1)
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()

        # Plot loss
        plt.subplot(2, 2, 2)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()

        # Plot precision
        plt.subplot(2, 2, 3)
        plt.plot(history.history['precision'], label='Train')
        plt.plot(history.history['val_precision'], label='Validation')
        plt.title('Model Precision')
        plt.ylabel('Precision')
        plt.xlabel('Epoch')
        plt.legend()

        # Plot recall
        plt.subplot(2, 2, 4)
        plt.plot(history.history['recall'], label='Train')
        plt.plot(history.history['val_recall'], label='Validation')
        plt.title('Model Recall')
        plt.ylabel('Recall')
        plt.xlabel('Epoch')
        plt.legend()

        plt.tight_layout()
        plt.savefig('results/training/training_history.png')
        logger.info(
            "Training history plot saved to results/training/training_history.png")
    except Exception as e:
        logger.warning(f"Could not plot training history: {str(e)}")


def plot_roc_curve(classifier, X_val, y_val):
    """Plot ROC curves for all models"""
    try:
        plt.figure(figsize=(10, 8))

        # Neural Network ROC
        y_pred_nn = classifier.model.predict(X_val)
        fpr_nn, tpr_nn, _ = roc_curve(y_val, y_pred_nn)
        roc_auc_nn = auc(fpr_nn, tpr_nn)
        plt.plot(fpr_nn, tpr_nn,
                 label=f'Neural Network (AUC = {roc_auc_nn:.3f})')

        # XGBoost ROC
        y_pred_xgb = classifier.xgb_model.predict_proba(X_val)[:, 1]
        fpr_xgb, tpr_xgb, _ = roc_curve(y_val, y_pred_xgb)
        roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
        plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_xgb:.3f})')

        # Random Forest ROC
        y_pred_rf = classifier.rf_model.predict_proba(X_val)[:, 1]
        fpr_rf, tpr_rf, _ = roc_curve(y_val, y_pred_rf)
        roc_auc_rf = auc(fpr_rf, tpr_rf)
        plt.plot(fpr_rf, tpr_rf,
                 label=f'Random Forest (AUC = {roc_auc_rf:.3f})')

        # Plot details
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)

        # Save the figure
        plt.savefig('results/training/roc_curves.png')
        logger.info("ROC curves saved to results/training/roc_curves.png")
    except Exception as e:
        logger.warning(f"Could not plot ROC curves: {str(e)}")


def plot_feature_importance(xgb_model, feature_names, top_n=20):
    """Plot feature importance from XGBoost model"""
    try:
        # Get feature importances
        importances = xgb_model.feature_importances_

        # Create DataFrame for easier handling
        features_df = pd.DataFrame({
            'Feature': feature_names[:len(importances)],
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        # Take top N features
        top_features = features_df.head(top_n)

        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=top_features)
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        plt.savefig('results/training/feature_importance.png')
        logger.info(
            "Feature importance plot saved to results/training/feature_importance.png")
    except Exception as e:
        logger.warning(f"Could not plot feature importance: {str(e)}")

# Function to directly analyze text without relying on preprocessor


def direct_analyze(classifier, text, vectorizer, scaler):
    """
    Direct analyze implementation that doesn't rely on preprocessor's scaler

    Args:
        classifier: ReviewClassifier instance
        text: Review text to analyze
        vectorizer: Trained TF-IDF vectorizer
        scaler: Trained StandardScaler

    Returns:
        dict: Analysis results
    """
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

    try:
        # Extract TF-IDF features
        X_tfidf = vectorizer.transform([text]).toarray()

        # Extract linguistic features
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        avg_word_length = char_count / max(1, word_count)

        # Lexical diversity
        unique_words = len(set(words))
        unique_ratio = unique_words / max(1, word_count)

        # Punctuation
        punct_count = len(re.findall(r'[^\w\s]', text))

        # Sentence structure
        sentences = text.split('.')
        sentence_count = len(sentences)
        avg_sentence_length = word_count / max(1, sentence_count)

        # Capitalization
        caps_count = sum(1 for c in text if c.isupper())
        caps_ratio = caps_count / max(1, len(text))

        # Stopwords
        stopword_count = sum(1 for w in words if w.lower() in STOPWORDS)
        stopwords_ratio = stopword_count / max(1, word_count)

        # Repetition
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        repetition = sum(count for count in word_counts.values() if count > 1)
        repetition_score = repetition / max(1, word_count)

        # Text complexity using TextBlob
        blob = TextBlob(text)
        grammar_complexity = 0
        for sentence in blob.sentences:
            grammar_complexity += len(sentence.tags)
        grammar_complexity /= max(1, len(blob.sentences))

        # Create numerical features array
        numerical_features = np.array([
            char_count,          # length
            word_count,          # word_count
            avg_word_length,     # avg_word_length
            unique_ratio,        # unique_words_ratio
            punct_count,         # punctuation_count
            sentence_count,      # sentence_count
            caps_ratio,          # caps_ratio
            stopwords_ratio,     # stopwords_ratio
            avg_sentence_length,  # avg_sentence_length
            repetition_score,    # repetition_score
            grammar_complexity   # grammar_complexity
        ]).reshape(1, -1)

        # Scale numerical features
        X_num = scaler.transform(numerical_features)

        # Combine features
        X_combined = np.hstack((X_tfidf, X_num))

        # Get predictions from each model
        nn_pred = float(classifier.model.predict(X_combined)[0][0])
        xgb_pred = float(classifier.xgb_model.predict_proba(X_combined)[0][1])
        rf_pred = float(classifier.rf_model.predict_proba(X_combined)[0][1])

        # Ensemble prediction
        ensemble_pred = (nn_pred + xgb_pred + rf_pred) / 3

        # Determine prediction
        prediction = 'CG' if ensemble_pred > 0.5 else 'OR'

        # Generate reasoning
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
                reasoning.append("Pattern matches computer-generated text")
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
    except Exception as e:
        logger.error(f"Error in direct analysis: {str(e)}")
        return {
            'prediction': 'Error',
            'confidence': 0.0,
            'reasoning': [f"Error: {str(e)}"]
        }


# ======================================================================
# MODEL TESTING AND EVALUATION
# ======================================================================

def test_model(test_data_path=None):
    """Test the model on test data or Amazon scraper data"""
    try:
        logger.info("Loading models and preprocessing components...")

        # Load the model components
        if os.path.exists('models/nn_model.keras'):
            nn_model = tf.keras.models.load_model('models/nn_model.keras')
            logger.info("Neural network model loaded")
        else:
            logger.error(
                "Neural network model not found. Please train the model first.")
            return

        if os.path.exists('models/xgb_model.joblib'):
            xgb_model = joblib.load('models/xgb_model.joblib')
            logger.info("XGBoost model loaded")
        else:
            logger.error("XGBoost model not found")
            return

        if os.path.exists('models/rf_model.joblib'):
            rf_model = joblib.load('models/rf_model.joblib')
            logger.info("Random Forest model loaded")
        else:
            logger.error("Random Forest model not found")
            return

        # Load the vectorizer and scaler
        if os.path.exists('models/vectorizer.joblib'):
            vectorizer = joblib.load('models/vectorizer.joblib')
            logger.info("Vectorizer loaded")
        else:
            logger.error("Vectorizer not found")
            return

        if os.path.exists('models/scaler.joblib'):
            scaler = joblib.load('models/scaler.joblib')
            logger.info("Scaler loaded")
        else:
            logger.error("Scaler not found")
            return

        # Determine test data source
        if test_data_path is None:
            # Try to use cached test path from training
            if 'test_path' in _test_data_cache:
                test_data_path = _test_data_cache['test_path']
                logger.info(f"Using cached test data path: {test_data_path}")
            else:
                # Look for reserved test data
                reserved_path = 'Data/Testing_data/reserved_test_data.csv'
                if os.path.exists(reserved_path):
                    test_data_path = reserved_path
                    logger.info(f"Using reserved test data: {test_data_path}")
                else:
                    # Look for Amazon scraper data
                    scraper_path = 'Data/Scraped_data/Test_data.csv'
                    if os.path.exists(scraper_path):
                        test_data_path = scraper_path
                        logger.info(
                            f"Using Amazon scraper data: {test_data_path}")
                    else:
                        logger.error(
                            "No test data found. Please provide a test data path.")
                        return

        # Load test data
        logger.info(f"Loading test data from {test_data_path}")
        test_df = pd.read_csv(test_data_path)

        # Determine the column containing review text
        text_column = None
        for col in ['review_text', 'text', 'review']:
            if col in test_df.columns:
                text_column = col
                break

        if text_column is None:
            logger.error(
                f"Could not find review text column in {list(test_df.columns)}")
            return

        logger.info(f"Using '{text_column}' column for review text")

        # Create a preprocessor instance
        with open('models/feature_config.json', 'r') as f:
            feature_config = json.load(f)

        preprocessor = DataPreprocessor(
            max_features=feature_config['max_features'])
        preprocessor.vectorizer = vectorizer
        preprocessor.scaler = scaler

        # Create classifier instance
        classifier = ReviewClassifier()
        classifier.model = nn_model
        classifier.xgb_model = xgb_model
        classifier.rf_model = rf_model

        # Process and predict
        results = []
        for idx, row in test_df.iterrows():
            review_text = row[text_column]
            if not isinstance(review_text, str) or len(review_text.strip()) == 0:
                continue

            # Use the direct_analyze function instead of relying on preprocessor
            analysis = direct_analyze(
                classifier, review_text, vectorizer, scaler)

            # Store results
            result_row = row.to_dict()
            result_row.update(analysis)
            results.append(result_row)

            # Log progress
            if idx % 10 == 0:
                logger.info(f"Processed {idx}/{len(test_df)} reviews")

        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv('results/testing/test_results.csv', index=False)
        logger.info(f"Results saved to results/testing/test_results.csv")

        # If labeled data is available, evaluate performance
        if 'label' in results_df.columns:
            y_true = (results_df['label'] == 'CG').astype(int)
            y_pred = (results_df['prediction'] == 'CG').astype(int)

            accuracy = np.mean(y_true == y_pred)

            # Plot confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['Original', 'Computer Generated'],
                        yticklabels=['Original', 'Computer Generated'])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix (Accuracy: {accuracy:.4f})')
            plt.savefig('results/testing/confusion_matrix.png')
            logger.info(
                f"Confusion matrix saved to results/testing/confusion_matrix.png")

            logger.info(f"Test Accuracy: {accuracy:.4f}")

        return results_df

    except Exception as e:
        logger.error(f"Error in testing model: {str(e)}")
        raise

# ======================================================================
# PHASE 2: REAL-TIME DATA PROCESSING USING APACHE KAFKA
# ======================================================================


def run_kafka_pipeline(nn_model, xgb_model, rf_model, vectorizer):
    """Start the Kafka consumer to process reviews in real-time"""
    try:
        logger.info("Starting Kafka consumer for real-time review analysis...")

        # Load the scaler
        if os.path.exists('models/scaler.joblib'):
            scaler = joblib.load('models/scaler.joblib')
        else:
            logger.error("Scaler not found")
            return

        # Create a consumer instance with all necessary models
        consumer = AdvancedReviewConsumer(
            model={'nn': nn_model, 'xgb': xgb_model, 'rf': rf_model},
            vectorizer=vectorizer,
            preprocessor=DataPreprocessor(max_features=vectorizer.max_features)
        )

        # Start processing messages
        logger.info("Kafka consumer started. Waiting for messages...")
        consumer.process_reviews()

    except KeyboardInterrupt:
        logger.info("Kafka consumer stopped by user")
    except Exception as e:
        logger.error(f"Error in Kafka pipeline: {str(e)}")

# ======================================================================
# MAIN ENTRY POINT
# ======================================================================


def main():
    """Main entry point with argument parsing for different modes"""
    parser = argparse.ArgumentParser(description='Amazon Review Detector CLI')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'serve', 'kafka-pipeline'], default='train',
                        help='Operation mode: train, test, serve, or kafka-pipeline (default: train)')
    parser.add_argument('--dataset', type=str, default='Data/Training_data/combined_dataset.csv',
                        help='Path to combined dataset (default: Data/Training_data/combined_dataset.csv)')
    parser.add_argument('--run-scraper', action='store_true',
                        help='Run Amazon scraper to collect new review data')
    parser.add_argument('--convert-only', action='store_true',
                        help='Only convert scraped JSON to CSV without collecting new data')

    args = parser.parse_args()

    # Handle different operation modes
    if args.mode == 'train':
        logger.info(f"Starting model training with dataset: {args.dataset}")
        classifier, vectorizer = train_model(args.dataset)
        logger.info("Model training completed")

    elif args.mode == 'test':
        if args.run_scraper:
            logger.info(
                "Running Amazon scraper to collect new data for testing")
            if run_amazon_scraper(args.convert_only):
                logger.info("Amazon scraper completed successfully")
                test_model('Data/Scraped_data/Test_data.csv')
            else:
                logger.error("Amazon scraper failed")
        else:
            logger.info("Testing model with existing data")
            test_model()

    elif args.mode == 'serve':
        logger.info("Starting real-time review analysis service")

        # Load models
        if os.path.exists('models/nn_model.keras'):
            nn_model = tf.keras.models.load_model('models/nn_model.keras')
        else:
            logger.error(
                "Neural network model not found. Please train the model first.")
            return

        if os.path.exists('models/xgb_model.joblib'):
            xgb_model = joblib.load('models/xgb_model.joblib')
        else:
            logger.error("XGBoost model not found")
            return

        if os.path.exists('models/rf_model.joblib'):
            rf_model = joblib.load('models/rf_model.joblib')
        else:
            logger.error("Random Forest model not found")
            return

        if os.path.exists('models/vectorizer.joblib'):
            vectorizer = joblib.load('models/vectorizer.joblib')
        else:
            logger.error("Vectorizer not found")
            return

        # Run Kafka pipeline
        run_kafka_pipeline(nn_model, xgb_model, rf_model, vectorizer)

    elif args.mode == 'kafka-pipeline':
        logger.info("Starting full Kafka pipeline with Amazon scraper")

        # Load models
        if os.path.exists('models/nn_model.keras'):
            nn_model = tf.keras.models.load_model('models/nn_model.keras')
        else:
            logger.error(
                "Neural network model not found. Please train the model first.")
            return

        if os.path.exists('models/xgb_model.joblib'):
            xgb_model = joblib.load('models/xgb_model.joblib')
        else:
            logger.error("XGBoost model not found")
            return

        if os.path.exists('models/rf_model.joblib'):
            rf_model = joblib.load('models/rf_model.joblib')
        else:
            logger.error("Random Forest model not found")
            return

        if os.path.exists('models/vectorizer.joblib'):
            vectorizer = joblib.load('models/vectorizer.joblib')
        else:
            logger.error("Vectorizer not found")
            return

        # First start the consumer in a separate process
        import multiprocessing
        consumer_process = multiprocessing.Process(
            target=run_kafka_pipeline,
            args=(nn_model, xgb_model, rf_model, vectorizer)
        )
        consumer_process.start()
        logger.info("Kafka consumer started in a separate process")

        # Give the consumer a moment to initialize
        time.sleep(5)

        # Then run the scraper and send data to Kafka
        if scraper_to_kafka():
            logger.info("Amazon scraper completed and sent data to Kafka")
        else:
            logger.error("Error in Amazon scraper to Kafka pipeline")

        # Keep the consumer running to process all messages
        logger.info("Consumer process is running. Press Ctrl+C to stop.")
        try:
            consumer_process.join()
        except KeyboardInterrupt:
            logger.info("Stopping Kafka pipeline...")
            consumer_process.terminate()
            consumer_process.join()


if __name__ == "__main__":
    main()

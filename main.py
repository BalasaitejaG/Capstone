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
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
import xgboost as xgb
import itertools

# Import custom modules
from src.processing.data_preprocessing import DataPreprocessor, ReviewClassifier
from src.kafka.consumer import AdvancedReviewConsumer
from src.utils.logger import setup_logger
from src.model_training.ml_models import MLModels


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


def train_model(dataset_path):
    """Train the model using dataset with 80% for training and 20% for testing"""
    try:
        # Load dataset with Latin-1 encoding which is more permissive
        dataset_df = pd.read_csv(dataset_path, encoding='latin1')
        logger.info(f"Loaded dataset with {len(dataset_df)} samples")

        # Check if 'label' column exists, if not create it based on available fields
        if 'label' not in dataset_df.columns:
            logger.info(
                "No 'label' column found in dataset. Attempting to create labels...")

            if 'review_sentiment' in dataset_df.columns:
                # Use wider thresholds for sentiment-based labeling
                dataset_df['review_sentiment'] = dataset_df['review_sentiment'].astype(
                    float)
                # Widened thresholds to include more data
                dataset_df['label'] = dataset_df['review_sentiment'].apply(
                    lambda x: 'CG' if x < 0.4 else ('OR' if x > 0.6 else None)
                )
                # Remove uncertain labels (those between 0.4 and 0.6)
                dataset_df = dataset_df[dataset_df['label'].notna()].copy()
                logger.info(
                    f"Created 'label' column using review_sentiment with wider thresholds. Remaining samples: {len(dataset_df)}")
            elif 'ratings' in dataset_df.columns:
                # Use a more inclusive approach for ratings
                dataset_df['ratings'] = pd.to_numeric(
                    dataset_df['ratings'], errors='coerce')
                # Widen criteria to include 3-star reviews
                dataset_df['label'] = dataset_df['ratings'].apply(
                    lambda x: 'CG' if x <= 2.5 else (
                        'OR' if x >= 3.5 else None)
                )
                # Only remove truly uncertain labels
                dataset_df = dataset_df[dataset_df['label'].notna()].copy()
                logger.info(
                    f"Created 'label' column using ratings with wider criteria. Remaining samples: {len(dataset_df)}")
            else:
                logger.error(
                    "No appropriate column found for creating labels. Cannot proceed.")
                raise ValueError(
                    "Cannot create labels from available data. Training requires labeled data.")

        # Verify we have enough data to continue
        if len(dataset_df) < 100:
            logger.error(
                f"Not enough data to train model: only {len(dataset_df)} samples after processing")
            raise ValueError(
                "Insufficient data for training. Need at least 100 samples.")

        # Make sure review column is properly identified
        if 'review_text' not in dataset_df.columns and 'review' in dataset_df.columns:
            dataset_df['review_text'] = dataset_df['review']
            logger.info("Using 'review' column as 'review_text'")

        # Drop rows with missing review text
        if 'review_text' in dataset_df.columns:
            dataset_df = dataset_df.dropna(subset=['review_text'])
            logger.info(
                f"After dropping rows with missing reviews: {len(dataset_df)} samples")

        # Separate labeled and unlabeled data
        labeled_df = dataset_df[dataset_df['label'].notna()].copy()
        unlabeled_df = dataset_df[dataset_df['label'].isna()].copy()

        logger.info(
            f"Found {len(labeled_df)} labeled and {len(unlabeled_df)} unlabeled samples")

        # Process unlabeled data with pseudo-labeling if we have enough labeled data
        if len(labeled_df) >= 300 and len(unlabeled_df) > 0:
            logger.info("Applying pseudo-labeling to unlabeled data...")
            # Initialize preprocessor for pseudo-labeling
            preprocessor = DataPreprocessor(max_features=1000)

            # Process dataset to extract features
            unlabeled_df = preprocessor._process_dataset(unlabeled_df)

            # Apply pseudo-labeling
            pseudo_labeled_df = preprocessor._pseudo_label_data(unlabeled_df)
            logger.info(
                f"Generated {len(pseudo_labeled_df)} pseudo-labeled samples")

            # Combine with labeled data (if any pseudo-labeled samples were generated)
            if len(pseudo_labeled_df) > 0:
                dataset_df = pd.concat(
                    [labeled_df, pseudo_labeled_df], ignore_index=True)
                logger.info(
                    f"Combined dataset now has {len(dataset_df)} samples")
            else:
                dataset_df = labeled_df
                logger.info(
                    "No reliable pseudo-labels generated, using only labeled data")
        else:
            dataset_df = labeled_df
            if len(unlabeled_df) > 0:
                logger.info(
                    "Not enough labeled data for reliable pseudo-labeling, using only labeled data")

        # Apply class balancing BEFORE splitting the data
        # Check class distribution
        class_counts = dataset_df['label'].value_counts()
        logger.info(f"Original class distribution: {class_counts.to_dict()}")

        # Balance classes if needed
        majority_class = class_counts.idxmax()
        minority_class = class_counts.idxmin()
        imbalance_ratio = class_counts[majority_class] / \
            class_counts[minority_class]

        if imbalance_ratio > 3:  # If imbalance is greater than 3:1
            logger.info(
                f"Detected class imbalance of {imbalance_ratio:.2f}:1. Applying balancing...")

            # Under-sample the majority class
            majority_samples = dataset_df[dataset_df['label']
                                          == majority_class]
            minority_samples = dataset_df[dataset_df['label']
                                          == minority_class]

            # Keep only 2x the minority class samples from the majority class
            balanced_majority = majority_samples.sample(
                n=min(len(minority_samples) * 2, len(majority_samples)),
                random_state=42
            )

            # Combine to create balanced dataset
            dataset_df = pd.concat([balanced_majority, minority_samples])
            logger.info(
                f"After balancing, class distribution: {dataset_df['label'].value_counts().to_dict()}")

        # Split data: 80% for training, 20% for testing
        train_df, test_df = train_test_split(
            dataset_df,
            test_size=0.2,
            random_state=42,
            stratify=dataset_df['label'] if 'label' in dataset_df.columns else None
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

        # Initialize preprocessor with reduced features
        preprocessor = DataPreprocessor(
            max_features=1000)  # Reduced from 5000 to 1000

        # Use the preprocess_dataset method with improved label handling
        X_train, X_val, y_train, y_val = preprocessor.preprocess_dataset(
            train_path)

        # Save the vectorizer and scaler
        joblib.dump(preprocessor.vectorizer, 'models/vectorizer.joblib')
        joblib.dump(preprocessor.scaler, 'models/scaler.joblib')
        with open('models/feature_config.json', 'w') as f:
            json.dump({'max_features': preprocessor.max_features}, f)

        # Build and train the models with improved regularization
        classifier = ReviewClassifier()
        classifier.build_model(input_shape=X_train.shape[1])
        history = classifier.train(X_train, y_train, X_val, y_val)

        # Save the trained models
        classifier.model.save('models/nn_model.keras')
        joblib.dump(classifier.xgb_model, 'models/xgb_model.joblib')
        joblib.dump(classifier.rf_model, 'models/rf_model.joblib')

        # Train additional ML models (SVM, KNN, Logistic Regression, Naive Bayes)
        logger.info(
            "Training additional ML models (SVM, KNN, Logistic Regression, Naive Bayes)...")
        ml_models = MLModels()
        ml_results = ml_models.train_models(X_train, y_train, X_val, y_val)

        # Plot ML model results including BERT
        ml_models.plot_all_results(ml_results, X_val, y_val)

        # Plot confusion matrices for all models in a single grid
        plot_all_model_confusion_matrices(classifier, ml_models, X_val, y_val)

        # Plot training history
        plot_training_history(history)

        # Evaluate model on validation set
        evaluation = classifier.evaluate(X_val, y_val)
        logger.info(f"Model Evaluation Results:")
        logger.info(f"Accuracy: {evaluation['accuracy']:.4f}")
        logger.info(f"Precision: {evaluation['precision']:.4f}")
        logger.info(f"Recall: {evaluation['recall']:.4f}")
        logger.info(f"F1 Score: {evaluation['f1']:.4f}")

        # Save the optimal threshold for future use
        with open('models/threshold_config.json', 'w') as f:
            json.dump({'optimal_threshold': evaluation['threshold']}, f)
        logger.info(
            f"Optimal threshold {evaluation['threshold']:.4f} saved for future predictions")

        # Generate ROC curve
        plot_roc_curve(classifier, X_val, y_val)

        # Create feature importance analysis if possible
        try:
            plot_feature_importance(classifier, X_val)
        except Exception as e:
            logger.warning(f"Could not plot feature importance: {str(e)}")

        # Perform cross-validation for more robust evaluation
        try:
            logger.info(
                "Performing 10-fold stratified cross-validation for robust evaluation...")
            from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate

            # Create a pipeline for cross-validation
            X = np.vstack([X_train, X_val])
            y = np.concatenate([y_train, y_val])

            # Create a CV-specific XGBoost model with balanced parameters
            cv_xgb_model = xgb.XGBClassifier(
                learning_rate=0.03,
                n_estimators=120,
                max_depth=4,
                min_child_weight=3,
                gamma=0.2,
                subsample=0.7,
                colsample_bytree=0.7,
                objective='binary:logistic',
                scale_pos_weight=1.2,
                tree_method='hist',
                reg_alpha=0.1,
                reg_lambda=1.0,
                eval_metric='auc',
                random_state=42
            )

            # Define the stratified k-fold with shuffling for stability
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

            # Use cross_validate to get multiple metrics at once
            scoring = {
                'accuracy': 'accuracy',
                'precision': 'precision',
                'recall': 'recall',
                'f1': 'f1',
                'roc_auc': 'roc_auc',
                'balanced_accuracy': 'balanced_accuracy'
            }

            cv_results = cross_validate(
                cv_xgb_model, X, y, cv=cv, scoring=scoring, return_train_score=False
            )

            # Calculate and log mean and standard deviation for each metric
            for metric in scoring.keys():
                scores = cv_results[f'test_{metric}']
                logger.info(
                    f"10-fold CV {metric}: {scores.mean():.4f} ± {scores.std():.4f}")

            # Calculate average ratio of precision to recall to check balance
            precision_scores = cv_results['test_precision']
            recall_scores = cv_results['test_recall']
            pr_ratios = precision_scores / recall_scores

            logger.info(
                f"Average precision/recall ratio: {pr_ratios.mean():.4f}")
            if pr_ratios.mean() > 1.3:
                logger.info(
                    "Note: Precision is significantly higher than recall. Consider adjusting threshold or class weights.")
            elif pr_ratios.mean() < 0.77:
                logger.info(
                    "Note: Recall is significantly higher than precision. Consider adjusting threshold or class weights.")
            else:
                logger.info("Precision and recall are reasonably balanced.")
        except Exception as e:
            logger.warning(f"Could not perform cross-validation: {str(e)}")

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
    """
    Plot training history with IEEE paper formatting

    Args:
        history: Training history from model.fit()
    """
    # Set up the matplotlib figure with appropriate dimensions for IEEE papers
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Create epochs array for x-axis
    epochs = range(1, len(history.history['loss']) + 1)

    # Plot loss curves with enhanced styling
    ax1.plot(epochs, history.history['loss'], 'o-', color='#0072B2', linewidth=2,
             label='Training Loss', markersize=4)
    ax1.plot(epochs, history.history['val_loss'], 's-', color='#D55E00', linewidth=2,
             label='Validation Loss', markersize=4)

    # Add IEEE-style formatting to loss plot
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlabel('Epochs', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper right', fontsize=10, frameon=True,
               facecolor='white', edgecolor='gray')

    # Find the epoch with minimum validation loss and mark it
    min_loss_epoch = np.argmin(history.history['val_loss']) + 1
    min_loss = history.history['val_loss'][min_loss_epoch - 1]
    ax1.annotate(f'Best: {min_loss:.4f}',
                 xy=(min_loss_epoch, min_loss),
                 xytext=(min_loss_epoch + 1, min_loss + 0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05,
                                 width=1.5, headwidth=8),
                 fontsize=10, fontweight='bold')

    # Plot accuracy curves with enhanced styling
    ax2.plot(epochs, history.history['accuracy'], 'o-', color='#0072B2', linewidth=2,
             label='Training Accuracy', markersize=4)
    ax2.plot(epochs, history.history['val_accuracy'], 's-', color='#D55E00', linewidth=2,
             label='Validation Accuracy', markersize=4)

    # Add IEEE-style formatting to accuracy plot
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlabel('Epochs', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='lower right', fontsize=10, frameon=True,
               facecolor='white', edgecolor='gray')

    # Find the epoch with maximum validation accuracy and mark it
    max_acc_epoch = np.argmax(history.history['val_accuracy']) + 1
    max_acc = history.history['val_accuracy'][max_acc_epoch - 1]
    ax2.annotate(f'Best: {max_acc:.4f}',
                 xy=(max_acc_epoch, max_acc),
                 xytext=(max_acc_epoch + 1, max_acc - 0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05,
                                 width=1.5, headwidth=8),
                 fontsize=10, fontweight='bold')

    # Ensure tight layout to avoid label cutoff
    plt.tight_layout()

    # Save the figure with high resolution for publication quality
    plt.savefig('results/training/training_history_ieee.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(
        "IEEE-formatted training history plot saved to results/training/training_history_ieee.png")

    # Export training history data to CSV for reference
    history_df = pd.DataFrame({
        'epoch': epochs,
        'train_loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'train_accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy']
    })

    history_df.to_csv('results/training/training_history.csv', index=False)
    logger.info(
        "Training history data saved to results/training/training_history.csv")


def plot_roc_curve(classifier, X_val, y_val):
    """Plot ROC curves for all models with IEEE paper formatting"""
    try:
        # Colorblind-friendly colors
        colors = [
            '#1f77b4',  # Blue
            '#ff7f0e',  # Orange
            '#2ca02c',  # Green
            '#d62728',  # Red
            '#9467bd',  # Purple
            '#8c564b',  # Brown
            '#e377c2',  # Pink
            '#7f7f7f'   # Gray
        ]

        # Line styles for better distinction
        line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']

        # Markers for better distinction
        markers = ['o', 's', '^', 'd', '*', 'x', '+', '.']

        plt.figure(figsize=(10, 8))

        # Neural Network ROC
        y_pred_nn = classifier.model.predict(X_val)
        fpr_nn, tpr_nn, _ = roc_curve(y_val, y_pred_nn)
        roc_auc_nn = auc(fpr_nn, tpr_nn)
        plt.plot(fpr_nn, tpr_nn, color=colors[0], linestyle=line_styles[0],
                 linewidth=2, marker=markers[0], markevery=0.1, markersize=5,
                 label=f'Neural Network (AUC = {roc_auc_nn:.3f})')

        # XGBoost ROC
        y_pred_xgb = classifier.xgb_model.predict_proba(X_val)[:, 1]
        fpr_xgb, tpr_xgb, _ = roc_curve(y_val, y_pred_xgb)
        roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
        plt.plot(fpr_xgb, tpr_xgb, color=colors[1], linestyle=line_styles[1],
                 linewidth=2, marker=markers[1], markevery=0.1, markersize=5,
                 label=f'XGBoost (AUC = {roc_auc_xgb:.3f})')

        # Random Forest ROC
        y_pred_rf = classifier.rf_model.predict_proba(X_val)[:, 1]
        fpr_rf, tpr_rf, _ = roc_curve(y_val, y_pred_rf)
        roc_auc_rf = auc(fpr_rf, tpr_rf)
        plt.plot(fpr_rf, tpr_rf, color=colors[2], linestyle=line_styles[2],
                 linewidth=2, marker=markers[2], markevery=0.1, markersize=5,
                 label=f'Random Forest (AUC = {roc_auc_rf:.3f})')

        # Check for additional ML models
        ml_models_dir = 'models'
        ml_model_files = [f for f in os.listdir(ml_models_dir) if f.endswith(
            '_model.joblib') and f not in ['xgb_model.joblib', 'rf_model.joblib']]

        for i, model_file in enumerate(ml_model_files):
            try:
                model_name = model_file.replace('_model.joblib', '')
                model = joblib.load(os.path.join(ml_models_dir, model_file))

                # Get predictions and plot ROC curve
                y_pred_ml = model.predict_proba(X_val)[:, 1]
                fpr_ml, tpr_ml, _ = roc_curve(y_val, y_pred_ml)
                roc_auc_ml = auc(fpr_ml, tpr_ml)

                # Convert model name for display (svm -> SVM, knn -> KNN, etc.)
                display_name = model_name.upper() if model_name in [
                    'svm', 'knn'] else model_name.replace('_', ' ').title()

                # Prevent index out of range
                color_idx = min(i + 3, len(colors) - 1)
                style_idx = min(i + 3, len(line_styles) - 1)
                marker_idx = min(i + 3, len(markers) - 1)

                plt.plot(fpr_ml, tpr_ml, color=colors[color_idx],
                         linestyle=line_styles[style_idx], linewidth=2,
                         marker=markers[marker_idx], markevery=0.1, markersize=5,
                         label=f'{display_name} (AUC = {roc_auc_ml:.3f})')
                logger.info(f"Added {display_name} to ROC curve plot")
            except Exception as e:
                logger.warning(
                    f"Could not include {model_file} in ROC plot: {str(e)}")

        # Plot the diagonal line
        plt.plot([0, 1], [0, 1], 'k--', lw=1.5)

        # Plot details with IEEE formatting
        plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
        plt.title('Receiver Operating Characteristic (ROC) Curves',
                  fontsize=16, fontweight='bold')

        # Move legend to outside of plot for better visibility
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

        # Improve grid
        plt.grid(True, linestyle='--', alpha=0.5)

        # Set precise axis limits
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])

        # Add annotated points at some key locations for best-performing model
        # Find best model based on AUC
        best_auc = max(roc_auc_nn, roc_auc_xgb, roc_auc_rf)
        if best_auc == roc_auc_nn:
            best_fpr, best_tpr = fpr_nn, tpr_nn
            best_name = "Neural Network"
        elif best_auc == roc_auc_xgb:
            best_fpr, best_tpr = fpr_xgb, tpr_xgb
            best_name = "XGBoost"
        else:
            best_fpr, best_tpr = fpr_rf, tpr_rf
            best_name = "Random Forest"

        # Annotate optimal point (closest to top-left corner)
        dists = (best_fpr ** 2 + (1 - best_tpr) ** 2) ** 0.5
        optimal_idx = np.argmin(dists)

        plt.scatter(best_fpr[optimal_idx], best_tpr[optimal_idx],
                    s=100, facecolors='none', edgecolors='black', linewidth=2)
        plt.annotate(f'Optimal threshold\n({best_fpr[optimal_idx]:.2f}, {best_tpr[optimal_idx]:.2f})',
                     (best_fpr[optimal_idx], best_tpr[optimal_idx]),
                     xytext=(best_fpr[optimal_idx]+0.1,
                             best_tpr[optimal_idx]-0.15),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

        # Add tight layout and save with higher resolution
        plt.tight_layout()
        plt.savefig('results/training/roc_curves_ieee.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(
            "IEEE-formatted ROC curves saved to results/training/roc_curves_ieee.png")
    except Exception as e:
        logger.warning(f"Could not plot ROC curves: {str(e)}")


def plot_feature_importance(classifier, X_val):
    """Plot feature importance for XGBoost model with IEEE paper formatting"""
    try:
        # Get feature importances from XGBoost model
        feature_importance = classifier.xgb_model.feature_importances_
        feature_names = X_val.columns

        # Sort features by importance
        indices = np.argsort(feature_importance)[::-1]

        # Keep only top 15 features for better visualization
        indices = indices[:15]

        # Colorblind-friendly colors for bars
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(indices)))

        # Create plot with improved styling
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance (XGBoost)',
                  fontsize=16, fontweight='bold')

        # Create bars with gradient color
        bars = plt.barh(range(len(indices)), feature_importance[indices], align='center',
                        color=colors, edgecolor='black', linewidth=1.2)

        # Add value labels to bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.002, bar.get_y() + bar.get_height()/2,
                     f'{width:.3f}', ha='left', va='center',
                     fontweight='bold', fontsize=9)

        # Add feature labels
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices],
                   fontsize=12, fontweight='bold')

        # Enhance grid and other elements
        plt.xlabel('Importance Score', fontsize=14, fontweight='bold')
        plt.ylabel('Features', fontsize=14, fontweight='bold')
        plt.grid(axis='x', linestyle='--', alpha=0.6)

        # Add descriptive annotations
        top_feature = feature_names[indices[0]]
        top_importance = feature_importance[indices[0]]
        plt.annotate(f'Most important: {top_feature}\n(Score: {top_importance:.3f})',
                     xy=(top_importance, 0),
                     xytext=(top_importance*0.7, -1.5),
                     arrowprops=dict(facecolor='black',
                                     shrink=0.05, width=1.5),
                     fontsize=10, fontweight='bold')

        # Add tight layout to prevent label cutoff
        plt.tight_layout()

        # Save high-resolution image
        plt.savefig('results/training/feature_importance_ieee.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(
            "IEEE-formatted feature importance plot saved to results/training/feature_importance_ieee.png")

        # Create a detailed feature importance table for the report
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)

        # Save to CSV for reference
        importance_df.to_csv(
            'results/training/feature_importance.csv', index=False)
        logger.info(
            "Feature importance data saved to results/training/feature_importance.csv")

    except Exception as e:
        logger.warning(f"Could not plot feature importance: {str(e)}")

# Function to directly analyze text without relying on preprocessor


def direct_analyze(classifier, text, vectorizer, scaler):
    """
    Advanced analyze implementation that incorporates key characteristics from the x.md 
    recommendations for fake review detection.

    Args:
        classifier: ReviewClassifier instance
        text: Review text to analyze
        vectorizer: Trained TF-IDF vectorizer
        scaler: Trained StandardScaler

    Returns:
        dict: Analysis results with detailed reasoning
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
        # Load optimal threshold if available
        threshold = 0.7  # Default conservative threshold to reduce false positives
        try:
            with open('models/threshold_config.json', 'r') as f:
                config = json.load(f)
                if 'optimal_threshold' in config:
                    threshold = config['optimal_threshold']
                    # Add a safety margin to reduce false positives
                    threshold = min(0.85, threshold + 0.1)
        except:
            logger.info("Using default threshold of 0.7")

        # Extract TF-IDF features
        X_tfidf = vectorizer.transform([text]).toarray()

        # --- LINGUISTIC FEATURES (from x.md recommendations) ---

        # Basic text preprocessing
        text_lower = text.lower()
        words = text.split()
        word_count = len(words)
        char_count = len(text)

        # 1. Lexical diversity - key indicator per x.md
        unique_words = len(set(words))
        unique_ratio = unique_words / max(1, word_count)

        # 2. Sentence structure analysis
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        sentence_count = len(sentences)

        # Calculate sentence lengths
        if sentence_count > 0:
            sentence_lengths = [len(s.split()) for s in sentences]
            avg_sentence_length = sum(sentence_lengths) / sentence_count
            # Variation in sentence length (natural text varies more)
            sentence_length_variation = np.std(
                sentence_lengths) / max(1, np.mean(sentence_lengths))
        else:
            avg_sentence_length = 0
            sentence_length_variation = 0

        # 3. Analyze extreme language patterns (per x.md)
        extreme_words = ['amazing', 'awesome', 'worst',
                         'terrible', 'best', 'perfect', 'horrible']
        extreme_word_count = sum(
            1 for word in words if word.lower() in extreme_words)
        extreme_ratio = extreme_word_count / max(1, word_count)

        # 4. Unusual punctuation patterns
        exclamation_count = text.count('!')
        question_count = text.count('?')
        caps_count = sum(1 for c in text if c.isupper())
        caps_ratio = caps_count / max(1, len(text))

        # 5. Filler words and stopwords (often differ in fake reviews)
        stopword_count = sum(1 for w in words if w.lower() in STOPWORDS)
        stopwords_ratio = stopword_count / max(1, word_count)

        # 6. Repetition patterns (fake reviews often repeat words/phrases)
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        repeat_words = sum(
            1 for word, count in word_counts.items() if count > 2)
        repetition_score = repeat_words / max(1, word_count)

        # 7. Unusual patterns that may indicate AI generation
        unusual_patterns = 0

        # 7.1 Repeated punctuation
        if re.search(r'[!?]{3,}', text):
            unusual_patterns += 1

        # 7.2 Excessive capitalization
        if re.search(r'[A-Z]{5,}', text):
            unusual_patterns += 1

        # 7.3 Repeated phrases (per x.md - linguistic anomalies)
        if re.search(r'(\w+\s+){3,}\1', text, re.I):
            unusual_patterns += 2

        # 7.4 Formulaic expressions common in fake reviews
        formulaic_phrases = ['highly recommend', 'great product', 'excellent quality',
                             'very happy', 'definitely recommend', 'money well spent']
        formulaic_count = sum(
            1 for phrase in formulaic_phrases if phrase in text_lower)
        if formulaic_count >= 2:
            unusual_patterns += formulaic_count

        # 8. Grammar and sentiment analysis
        blob = TextBlob(text)

        # 8.1 Grammar complexity
        grammar_complexity = 0
        for sentence in blob.sentences:
            grammar_complexity += len(sentence.tags)
        grammar_complexity /= max(1, len(blob.sentences))

        # 8.2 Sentiment consistency (fake reviews often have inconsistent sentiment)
        sentences_sentiment = [
            sentence.sentiment.polarity for sentence in blob.sentences if len(sentence.words) > 3]
        if sentences_sentiment:
            sentiment_std = np.std(sentences_sentiment)
            avg_sentiment = np.mean(sentences_sentiment)
            # Higher is more consistent
            sentiment_consistency = 1 - min(1, sentiment_std)
        else:
            sentiment_consistency = 0.5
            avg_sentiment = 0

        # 9. Content specificity - fake reviews tend to be generic (per x.md)
        product_specific_words = ['use', 'quality',
                                  'size', 'color', 'price', 'material', 'feature']
        specific_detail_count = sum(
            1 for word in product_specific_words if word in text_lower)
        specific_details_ratio = specific_detail_count / \
            len(product_specific_words)

        # Create numerical features array
        numerical_features = np.array([
            char_count,                # length
            word_count,                # word_count
            avg_sentence_length,       # avg_sentence_length
            # unique_words_ratio (key feature from x.md)
            unique_ratio,
            sentence_count,            # number of sentences
            caps_ratio,                # uppercase ratio
            stopwords_ratio,           # stopwords frequency
            repetition_score,          # word repetition
            grammar_complexity,        # grammar complexity
            # sentiment consistency (x.md mentions this)
            sentiment_consistency,
            extreme_ratio,             # extreme language (x.md mentions this)
            # content specificity (x.md mentions this)
            specific_details_ratio
        ]).reshape(1, -1)

        # Scale only the original features the model expects (first 11)
        X_num = scaler.transform(numerical_features[:, :11])

        # Combine features
        X_combined = np.hstack((X_tfidf, X_num))

        # Get predictions from each model
        nn_pred = float(classifier.model.predict(X_combined)[0][0])
        xgb_pred = float(classifier.xgb_model.predict_proba(X_combined)[0][1])
        rf_pred = float(classifier.rf_model.predict_proba(X_combined)[0][1])

        # Ensemble prediction with weighted average
        ensemble_pred = 0.4 * nn_pred + 0.4 * xgb_pred + 0.2 * rf_pred

        # Determine prediction using the higher threshold for CG class
        prediction = 'CG' if ensemble_pred > threshold else 'OR'

        # Calculate confidence
        if prediction == 'CG':
            confidence = ensemble_pred
        else:
            confidence = 1 - ensemble_pred

        # Generate detailed reasoning based on x.md guidelines and linguistic patterns
        reasoning = []

        if prediction == 'CG':
            # Core linguistic indicators from x.md
            if unique_ratio < 0.55:
                reasoning.append(
                    "Low vocabulary diversity (AI-generated content typically has lower lexical diversity)")

            if repetition_score > 0.15:
                reasoning.append(
                    "High word repetition patterns characteristic of AI text")

            if stopwords_ratio < 0.15 or stopwords_ratio > 0.6:
                reasoning.append(
                    "Unusual stopword frequency compared to typical human writing")

            # Sentence structure indicators
            if sentence_length_variation < 0.3 and sentence_count > 2:
                reasoning.append(
                    "Suspiciously consistent sentence lengths (human writing typically varies more)")

            # Sentiment patterns
            if sentiment_consistency > 0.85 and abs(avg_sentiment) > 0.5:
                reasoning.append(
                    "Unusually consistent sentiment throughout (lacks natural variation)")

            # Unusual patterns detection
            if unusual_patterns > 0:
                reasoning.append(
                    f"Contains {unusual_patterns} unusual language patterns typical of AI-generated text")

            # Generic content detection
            if specific_details_ratio < 0.3 and word_count > 30:
                reasoning.append(
                    "Lacks specific product details that are common in genuine reviews")

            # Extreme language
            if extreme_ratio > 0.1:
                reasoning.append(
                    "Uses excessive superlatives and extreme language")

            if formulaic_count >= 2:
                reasoning.append(
                    f"Contains formulaic expressions commonly used in fake reviews")

            if len(reasoning) == 0:
                reasoning.append(
                    "Overall linguistic patterns match computer-generated text")
        else:
            # Linguistic indicators for human writing
            if unique_ratio > 0.7:
                reasoning.append(
                    "High vocabulary diversity typical of human writing")

            if repetition_score < 0.1:
                reasoning.append(
                    "Natural word usage patterns without excessive repetition")

            if sentence_length_variation > 0.4:
                reasoning.append("Natural variation in sentence structure")

            if 0.3 < sentiment_consistency < 0.7:
                reasoning.append(
                    "Natural sentiment variation throughout the text")

            if specific_details_ratio > 0.4:
                reasoning.append(
                    "Contains specific product details characteristic of genuine reviews")

            if len(reasoning) == 0:
                reasoning.append(
                    "Overall linguistic patterns match human-written text")

        return {
            'prediction': prediction,
            'confidence': float(confidence),
            'reasoning': reasoning
        }
    except Exception as e:
        logger.error(f"Error in advanced analysis: {str(e)}")
        return {
            'prediction': 'Error',
            'confidence': 0.0,
            'reasoning': [f"Error: {str(e)}"]
        }


# ======================================================================
# MODEL TESTING AND EVALUATION
# ======================================================================

def test_model(test_data_path=None):
    """Test the model with reserved test data or new scraped data"""
    try:
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

        if os.path.exists('models/scaler.joblib'):
            scaler = joblib.load('models/scaler.joblib')
        else:
            logger.error("Scaler not found")
            return

        # Load optimal threshold if available
        threshold = 0.7  # Default to a more conservative threshold
        if os.path.exists('models/threshold_config.json'):
            with open('models/threshold_config.json', 'r') as f:
                config = json.load(f)
                if 'optimal_threshold' in config:
                    threshold = config['optimal_threshold']
                    # Add safety margin to reduce false positives
                    threshold = min(0.85, threshold + 0.1)
            logger.info(f"Using threshold {threshold:.4f} for predictions")
        else:
            logger.info(
                f"Using conservative default threshold {threshold:.4f}")

        # If no test data path is provided, use the reserved test data
        if test_data_path is None:
            global _test_data_cache
            if 'test_path' in _test_data_cache:
                test_data_path = _test_data_cache['test_path']
            else:
                test_data_path = 'Data/Testing_data/reserved_test_data.csv'

        logger.info(f"Using test data from: {test_data_path}")

        # Load test data
        if os.path.exists(test_data_path):
            # Use latin1 encoding to handle special characters
            test_df = pd.read_csv(test_data_path, encoding='latin1')
            logger.info(f"Loaded {len(test_df)} test samples")
        else:
            logger.error(f"Test data file not found: {test_data_path}")
            return

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
        # Use the loaded threshold
        classifier.optimal_threshold = threshold

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

            # Calculate metrics
            accuracy = np.mean(y_true == y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

            logger.info(f"Test Metrics:")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1 Score: {f1:.4f}")

            # Plot confusion matrix
            plot_confusion_matrix(y_true, y_pred, ['CG', 'OR'])

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


def analyze_dataset_labels(dataset_path):
    """Analyze dataset labels for potential issues that could lead to overfitting"""
    try:
        # Load dataset
        dataset_df = pd.read_csv(dataset_path, encoding='latin1')
        logger.info(
            f"Analyzing labels in dataset with {len(dataset_df)} samples")

        # Check if label column exists
        if 'label' not in dataset_df.columns:
            logger.error("CRITICAL ISSUE: No 'label' column found in dataset.")
            logger.error("A label column is required for supervised learning.")

            # Check if we can create a label column from other data
            potential_label_sources = []
            if 'review_sentiment' in dataset_df.columns:
                potential_label_sources.append('review_sentiment')
            if 'ratings' in dataset_df.columns:
                potential_label_sources.append('ratings')

            if potential_label_sources:
                logger.info(
                    f"Potential sources for creating labels: {', '.join(potential_label_sources)}")
                logger.info(
                    "You can use the train mode to automatically create labels from these fields.")
            else:
                logger.error(
                    "No suitable columns found for creating labels. Manual labeling required.")

            return ["Missing label column"]

        # Class balance analysis
        class_counts = dataset_df['label'].value_counts()
        total = len(dataset_df)
        logger.info(f"Class distribution: {class_counts.to_dict()}")

        imbalance_ratio = class_counts.max() / class_counts.min()
        logger.info(f"Class imbalance ratio: {imbalance_ratio:.2f}")
        if imbalance_ratio > 3:
            logger.warning(
                f"Severe class imbalance detected (ratio: {imbalance_ratio:.2f}). Consider better sampling.")

        # Check for label correlation with metadata
        potential_correlations = []

        # Check correlation with review length
        if 'review' in dataset_df.columns:
            dataset_df['review_length'] = dataset_df['review'].str.len()
            length_by_class = dataset_df.groupby(
                'label')['review_length'].mean()
            length_ratio = length_by_class.max() / length_by_class.min()
            logger.info(
                f"Average review length by class: {length_by_class.to_dict()}")
            if length_ratio > 1.5:
                potential_correlations.append(
                    f"Review length differs by {length_ratio:.2f}x between classes")

        # Check correlation with ratings if available
        if 'ratings' in dataset_df.columns:
            dataset_df['ratings'] = pd.to_numeric(
                dataset_df['ratings'], errors='coerce')
            rating_by_class = dataset_df.groupby('label')['ratings'].mean()
            logger.info(
                f"Average rating by class: {rating_by_class.to_dict()}")
            if abs(rating_by_class.max() - rating_by_class.min()) > 1.0:
                potential_correlations.append(
                    "Strong correlation between ratings and labels")

        # Check correlation with sentiment if available
        if 'review_sentiment' in dataset_df.columns:
            sentiment_by_class = dataset_df.groupby(
                'label')['review_sentiment'].mean()
            logger.info(
                f"Average sentiment by class: {sentiment_by_class.to_dict()}")
            if abs(sentiment_by_class.max() - sentiment_by_class.min()) > 0.3:
                potential_correlations.append(
                    "Strong correlation between sentiment and labels")

        # Report findings
        if potential_correlations:
            logger.warning(
                "Potential dataset issues found that may cause overfitting:")
            for issue in potential_correlations:
                logger.warning(f"- {issue}")
            logger.warning(
                "Model may be learning these correlations rather than real patterns")

        return potential_correlations
    except Exception as e:
        logger.error(f"Error analyzing dataset labels: {str(e)}")
        return ["Error during analysis"]


def main():
    """Main entry point with argument parsing for different modes"""
    parser = argparse.ArgumentParser(description='Amazon Review Detector CLI')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'serve', 'kafka-pipeline', 'analyze'],
                        default='train',
                        help='Operation mode: train, test, serve, kafka-pipeline, or analyze (default: train)')
    parser.add_argument('--dataset', type=str, default='Data/final_dataset.csv',
                        help='Path to dataset file (default: Data/final_dataset.csv)')
    parser.add_argument('--run-scraper', action='store_true',
                        help='Run Amazon scraper to collect new review data')
    parser.add_argument('--convert-only', action='store_true',
                        help='Only convert scraped JSON to CSV without collecting new data')

    args = parser.parse_args()

    # Handle different operation modes
    if args.mode == 'train':
        # First analyze dataset for potential issues
        issues = analyze_dataset_labels(args.dataset)
        if "Missing label column" in issues:
            logger.info(
                "Proceeding with training will create labels automatically")

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

    elif args.mode == 'analyze':
        logger.info(f"Analyzing dataset: {args.dataset}")
        issues = analyze_dataset_labels(args.dataset)
        if not issues:
            logger.info("No issues found with dataset labels")
        else:
            logger.error(
                "Dataset analysis complete. Issues were found that need to be addressed.")
            for i, issue in enumerate(issues, 1):
                logger.error(f"Issue {i}: {issue}")


def plot_confusion_matrix(y_true, y_pred, labels):
    """
    Plot confusion matrix with IEEE paper formatting

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create figure with appropriate size for publication
    plt.figure(figsize=(10, 8))

    # Use a colorblind-friendly colormap
    cmap = plt.cm.Blues

    # Plot confusion matrix with enhanced styling
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    # Add class labels with improved typography
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45,
               ha='right', fontsize=12, fontweight='bold')
    plt.yticks(tick_marks, labels, fontsize=12, fontweight='bold')

    # Add a title with appropriate formatting
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')

    # Add text annotations to the cells
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=14, fontweight='bold')

    # Calculate and display metrics on the plot
    accuracy = np.trace(cm) / np.sum(cm)
    misclass = 1 - accuracy

    plt.text(cm.shape[1]-0.1, cm.shape[0]+0.3,
             f'Accuracy: {accuracy:.3f}',
             horizontalalignment='right',
             fontsize=12, fontweight='bold')

    plt.text(0.1, cm.shape[0]+0.3,
             f'Error Rate: {misclass:.3f}',
             horizontalalignment='left',
             fontsize=12, fontweight='bold')

    # Ensure tight layout to prevent cutoff
    plt.tight_layout()

    # Save high resolution image
    plt.savefig('results/training/confusion_matrix_ieee.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(
        "IEEE-formatted confusion matrix saved to results/training/confusion_matrix_ieee.png")

    # Calculate additional metrics for reporting
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Save metrics to a JSON file for reference
    metrics = {
        'accuracy': float(accuracy),
        'error_rate': float(misclass),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }

    with open('results/training/classification_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    logger.info(
        "Classification metrics saved to results/training/classification_metrics.json")


def plot_all_model_confusion_matrices(classifier, ml_models, X_val, y_val, labels=['CG', 'OR'], save_path='results/training/all_model_confusion_matrices.png'):
    """
    Plot confusion matrices for all models (neural network, XGBoost, random forest, and ML models) in a single grid

    Args:
        classifier: ReviewClassifier instance containing NN, XGBoost, and Random Forest models
        ml_models: MLModels instance with additional ML models
        X_val: Validation features
        y_val: Validation labels  
        labels: Class labels
        save_path: Path to save the figure
    """
    # Count the number of models
    n_models = 3 + len(ml_models.models)  # NN, XGB, RF + other ML models

    # Calculate grid dimensions (trying to make it as square as possible)
    grid_size = int(np.ceil(np.sqrt(n_models)))
    fig, axes = plt.subplots(grid_size, grid_size,
                             figsize=(grid_size * 4, grid_size * 4))

    # Flatten axes for easier indexing
    if grid_size > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # Set a global colormap for consistency
    cmap = plt.cm.Blues

    # Keep track of max value for colorbar normalization
    max_value = 0

    model_count = 0

    # Neural Network confusion matrix
    y_pred_nn = (classifier.model.predict(X_val) > 0.5).astype(int).flatten()
    cm_nn = confusion_matrix(y_val, y_pred_nn)
    max_value = max(max_value, np.max(cm_nn))
    sns.heatmap(cm_nn, annot=True, fmt='d', cmap=cmap,
                ax=axes[model_count], cbar=False)
    accuracy_nn = np.trace(cm_nn) / np.sum(cm_nn)
    axes[model_count].set_title(
        f"Neural Network\nAccuracy: {accuracy_nn:.3f}", fontweight='bold')
    axes[model_count].set_xlabel('Predicted', fontweight='bold')
    axes[model_count].set_ylabel('Actual', fontweight='bold')
    model_count += 1

    # XGBoost confusion matrix
    y_pred_xgb = (classifier.xgb_model.predict_proba(
        X_val)[:, 1] > 0.5).astype(int)
    cm_xgb = confusion_matrix(y_val, y_pred_xgb)
    max_value = max(max_value, np.max(cm_xgb))
    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap=cmap,
                ax=axes[model_count], cbar=False)
    accuracy_xgb = np.trace(cm_xgb) / np.sum(cm_xgb)
    axes[model_count].set_title(
        f"XGBoost\nAccuracy: {accuracy_xgb:.3f}", fontweight='bold')
    axes[model_count].set_xlabel('Predicted', fontweight='bold')
    axes[model_count].set_ylabel('Actual', fontweight='bold')
    model_count += 1

    # Random Forest confusion matrix
    y_pred_rf = (classifier.rf_model.predict_proba(
        X_val)[:, 1] > 0.5).astype(int)
    cm_rf = confusion_matrix(y_val, y_pred_rf)
    max_value = max(max_value, np.max(cm_rf))
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap=cmap,
                ax=axes[model_count], cbar=False)
    accuracy_rf = np.trace(cm_rf) / np.sum(cm_rf)
    axes[model_count].set_title(
        f"Random Forest\nAccuracy: {accuracy_rf:.3f}", fontweight='bold')
    axes[model_count].set_xlabel('Predicted', fontweight='bold')
    axes[model_count].set_ylabel('Actual', fontweight='bold')
    model_count += 1

    # Add confusion matrices for additional ML models
    for name, model in ml_models.models.items():
        if model_count < len(axes):
            ax = axes[model_count]

            # Get predictions
            y_pred = model.predict(X_val)

            # Calculate confusion matrix
            cm = confusion_matrix(y_val, y_pred)
            max_value = max(max_value, np.max(cm))

            # Plot
            sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax, cbar=False)

            # Add accuracy value
            accuracy = np.trace(cm) / np.sum(cm)

            # Make the title more descriptive with model name and accuracy
            display_name = name.upper() if name in [
                'svm', 'knn'] else name.replace('_', ' ').title()
            ax.set_title(
                f"{display_name}\nAccuracy: {accuracy:.3f}", fontweight='bold')
            ax.set_xlabel('Predicted', fontweight='bold')
            ax.set_ylabel('Actual', fontweight='bold')

            model_count += 1

    # Hide any unused subplots
    for i in range(model_count, len(axes)):
        axes[i].axis('off')

    # Add a colorbar to the right of the subplots
    fig.subplots_adjust(right=0.85, wspace=0.3, hspace=0.3)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])

    # Create a ScalarMappable for the colorbar
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=0, vmax=max_value)
    scalar_mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar_mappable.set_array([])
    fig.colorbar(scalar_mappable, cax=cbar_ax, label='Count')

    # Add a super title
    plt.suptitle('Confusion Matrices for All Models',
                 fontsize=20, fontweight='bold', y=0.98)

    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"All model confusion matrices saved to {save_path}")


if __name__ == "__main__":
    main()

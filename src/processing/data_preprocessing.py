from nltk.corpus import stopwords
from tensorflow.keras.layers import Input
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.logger import setup_logger
import re
import os
import random
from sklearn.preprocessing import StandardScaler
from nltk.tokenize import word_tokenize
import nltk
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from textblob import TextBlob
import xgboost as xgb
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from multiprocessing import Pool, cpu_count
from functools import partial
import time
from sklearn.cluster import KMeans
from scipy.sparse import hstack
import joblib

# Download required NLTK resources once at module level
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

logger = setup_logger(__name__)


class DataPreprocessor:
    def __init__(self, max_features=1000):  # Reduced from 3000 to 1000
        self.max_features = max_features
        # Update vectorizer settings with more conservative parameters
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Keep (1,2) for basic bi-grams
            min_df=5,  # Increased from 3 to filter more noise
            max_df=0.8,  # Decreased from 0.9 to remove more common words
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True
        )
        self.scaler = StandardScaler()
        self.tokenized_texts = {}  # Cache for tokenized texts
        self.start_time = None

    def preprocess_data(self, labeled_path, unlabeled_path):
        try:
            self.start_time = time.time()
            logger.info(
                "Loading and preprocessing both labeled and unlabeled datasets")

            # Load both datasets
            try:
                labeled_df = pd.read_csv(labeled_path)
                logger.info(f"Initial labeled data size: {len(labeled_df)}")
                # Validate required columns in labeled dataset
                if 'review_text' not in labeled_df.columns:
                    if 'review' in labeled_df.columns:
                        labeled_df['review_text'] = labeled_df['review']
                    else:
                        raise ValueError(
                            f"Missing required column 'review_text' or 'review' in {labeled_path}")

                if 'label' not in labeled_df.columns:
                    if 'rating' in labeled_df.columns:
                        # Convert ratings to labels (assuming low ratings < 3 might indicate fake reviews)
                        labeled_df['label'] = labeled_df['rating'].apply(
                            lambda x: 'CG' if float(x) < 3 else 'OR')
                    else:
                        logger.warning(
                            f"No label column found in {labeled_path}. Using all as 'OR' by default.")
                        labeled_df['label'] = 'OR'
            except Exception as e:
                logger.error(f"Error loading labeled dataset: {str(e)}")
                raise

            try:
                unlabeled_df = pd.read_csv(unlabeled_path)
                logger.info(
                    f"Initial unlabeled data size: {len(unlabeled_df)}")
                # Validate required columns in unlabeled dataset
                if 'review_text' not in unlabeled_df.columns:
                    if 'review' in unlabeled_df.columns:
                        unlabeled_df['review_text'] = unlabeled_df['review']
                    else:
                        raise ValueError(
                            f"Missing required column 'review_text' or 'review' in {unlabeled_path}")
            except Exception as e:
                logger.error(f"Error loading unlabeled dataset: {str(e)}")
                raise

            # Process labeled data with progress logging
            logger.info("Processing labeled data...")
            labeled_df = self._process_dataset(labeled_df)
            logger.info(
                f"Labeled data processing completed in {time.time() - self.start_time:.2f} seconds")

            logger.info("Processing unlabeled data...")
            unlabeled_df = self._process_dataset(unlabeled_df)
            logger.info(
                f"Unlabeled data processing completed in {time.time() - self.start_time:.2f} seconds")

            logger.info(
                f"After processing - labeled data size: {len(labeled_df)}")
            logger.info(
                f"After processing - unlabeled data size: {len(unlabeled_df)}")

            # Data augmentation for the minority class
            logger.info("Performing data augmentation for class balance...")
            labeled_df = self._augment_minority_class(labeled_df)
            logger.info(
                f"After augmentation - labeled data size: {len(labeled_df)}")

            # Use high-confidence predictions to label unlabeled data
            logger.info("Pseudo-labeling unlabeled data...")
            unlabeled_df = self._pseudo_label_data(unlabeled_df)
            logger.info(
                f"Pseudo-labeling completed in {time.time() - self.start_time:.2f} seconds")

            # Combine datasets
            combined_df = pd.concat(
                [labeled_df, unlabeled_df], ignore_index=True)

            # Split into training and validation with stratification
            logger.info("Splitting data into train and validation sets...")
            train_df, val_df = train_test_split(
                combined_df,
                test_size=0.2,
                random_state=42,
                stratify=combined_df['label']
            )

            # Process features with progress logging
            logger.info("Generating TF-IDF features...")
            start_tfidf = time.time()
            X_train_tfidf = self.vectorizer.fit_transform(train_df['text'])
            X_val_tfidf = self.vectorizer.transform(val_df['text'])
            logger.info(
                f"TF-IDF vectorization completed in {time.time() - start_tfidf:.2f} seconds")

            # Get the number of TF-IDF features
            n_tfidf_features = X_train_tfidf.shape[1]
            logger.info(f"Number of TF-IDF features: {n_tfidf_features}")

            numerical_features = [
                'length', 'word_count', 'avg_word_length', 'unique_words_ratio',
                'punctuation_count', 'sentence_count', 'caps_ratio', 'stopwords_ratio',
                'avg_sentence_length', 'repetition_score', 'grammar_complexity'
            ]

            logger.info("Scaling numerical features...")
            start_scaling = time.time()
            X_train_num = self.scaler.fit_transform(
                train_df[numerical_features])
            X_val_num = self.scaler.transform(val_df[numerical_features])
            logger.info(
                f"Feature scaling completed in {time.time() - start_scaling:.2f} seconds")

            # Calculate total features
            total_features = n_tfidf_features + len(numerical_features)
            logger.info(f"Total number of features: {total_features}")

            # Combine features
            logger.info("Combining features...")
            X_train = self._combine_features(X_train_tfidf, X_train_num)
            X_val = self._combine_features(X_val_tfidf, X_val_num)

            y_train = (train_df['label'] == 'CG').astype(int)
            y_val = (val_df['label'] == 'CG').astype(int)

            logger.info(
                f"All preprocessing completed in {time.time() - self.start_time:.2f} seconds")
            return X_train, X_val, y_train, y_val

        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def _process_dataset(self, df):
        """Process a single dataset with all features using vectorized operations where possible"""
        start_process = time.time()

        # Make a copy of the dataframe to avoid SettingWithCopyWarning
        df = df.copy()

        # Clean text
        if 'text' not in df.columns:
            df['text'] = df['review_text'].fillna('')

        # Basic filtering
        df = df[df['text'].str.len() > 10]  # Remove very short reviews

        # Use vectorized text cleaning
        logger.info(f"Cleaning text for {len(df)} reviews...")
        df.loc[:, 'text'] = df['text'].apply(self._clean_text)
        logger.info(
            f"Text cleaning completed in {time.time() - start_process:.2f} seconds")

        # Handle missing values in all columns
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df.loc[:, col] = df[col].fillna(0)
            else:
                df.loc[:, col] = df[col].fillna('')

        # Use multiprocessing for feature extraction if dataset is large
        if len(df) > 1000:
            logger.info(
                f"Using multiprocessing for feature extraction on {len(df)} reviews...")
            # Split dataframe into chunks for multiprocessing
            df_splits = np.array_split(df, min(cpu_count(), 8))

            with Pool(processes=min(cpu_count(), 8)) as pool:
                # Process each chunk in parallel
                results = pool.map(self._extract_features_chunk, df_splits)

            # Combine results
            df = pd.concat(results, ignore_index=True)
            logger.info(
                f"Multiprocessing feature extraction completed in {time.time() - start_process:.2f} seconds")
        else:
            # For smaller datasets, process sequentially
            logger.info(f"Extracting features for {len(df)} reviews...")
            df = self._extract_features_chunk(df)
            logger.info(
                f"Feature extraction completed in {time.time() - start_process:.2f} seconds")

        # Filter low-quality samples with more flexible thresholds
        df = df[
            (df['word_count'].between(5, 2000)) &
            (df['unique_words_ratio'] > 0.2) &
            (df['stopwords_ratio'] < 0.7) &
            (df['caps_ratio'] < 0.5) &
            (df['avg_sentence_length'] > 2)
        ]

        return df

    def _extract_features_chunk(self, df_chunk):
        """Extract features for a chunk of the dataframe - used in multiprocessing"""
        # Make a copy to avoid SettingWithCopyWarning
        df_chunk = df_chunk.copy()

        # Vectorized operations where possible
        df_chunk.loc[:, 'length'] = df_chunk['text'].str.len()

        # These operations need to be applied row by row
        df_chunk.loc[:, 'word_count'] = df_chunk['text'].apply(
            lambda x: len(x.split()))
        df_chunk.loc[:, 'sentence_count'] = df_chunk['text'].apply(
            lambda x: max(len(x.split('.')), 1))

        # Calculate text-based features
        for idx, row in df_chunk.iterrows():
            text = row['text']
            words = text.split()

            # Only tokenize once per text
            if len(words) > 0:
                df_chunk.loc[idx, 'avg_word_length'] = sum(
                    len(word) for word in words) / len(words)
                unique_words = set(words)
                df_chunk.loc[idx, 'unique_words_ratio'] = len(
                    unique_words) / len(words)
            else:
                df_chunk.loc[idx, 'avg_word_length'] = 0
                df_chunk.loc[idx, 'unique_words_ratio'] = 0

            df_chunk.loc[idx, 'punctuation_count'] = sum(
                1 for char in text if char in '.,!?')
            df_chunk.loc[idx, 'caps_ratio'] = sum(
                1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0

            # Calculate stopwords ratio
            if len(words) > 0:
                stopword_count = sum(
                    1 for word in words if word.lower() in STOPWORDS)
                df_chunk.loc[idx,
                             'stopwords_ratio'] = stopword_count / len(words)
            else:
                df_chunk.loc[idx, 'stopwords_ratio'] = 0

            # Calculate sentence-based metrics
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if sentences:
                sentence_word_counts = [len(s.split()) for s in sentences]
                df_chunk.loc[idx, 'avg_sentence_length'] = sum(
                    sentence_word_counts) / len(sentences)

                # Repetition score (simplified)
                word_freq = {}
                for word in words:
                    word_freq[word] = word_freq.get(word, 0) + 1
                repeat_words = sum(
                    1 for word, count in word_freq.items() if count > 2)
                df_chunk.loc[idx, 'repetition_score'] = repeat_words / \
                    len(words) if words else 0

                # Grammar complexity (simplified)
                complex_markers = ['although', 'however',
                                   'nevertheless', 'therefore', 'thus', 'moreover']
                complex_count = sum(any(marker in s.lower()
                                    for marker in complex_markers) for s in sentences)
                df_chunk.loc[idx, 'grammar_complexity'] = complex_count / \
                    len(sentences)
            else:
                df_chunk.loc[idx, 'avg_sentence_length'] = 0
                df_chunk.loc[idx, 'repetition_score'] = 0
                df_chunk.loc[idx, 'grammar_complexity'] = 0

        return df_chunk

    def _clean_text(self, text):
        """Enhanced text cleaning"""
        text = text.lower()
        text = re.sub(r'http\S+|www.\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+', '', text)  # Remove mentions
        text = re.sub(r'[^\w\s.,!?]', ' ', text)  # Keep basic punctuation
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        return text.strip()

    def _combine_features(self, tfidf_features, numerical_features):
        """Combine TF-IDF and numerical features more efficiently"""
        start = time.time()
        # If tfidf_features is already dense, no need to convert
        if isinstance(tfidf_features, np.ndarray):
            combined = np.hstack([tfidf_features, numerical_features])
        else:
            # Convert sparse to dense only when necessary
            combined = np.hstack(
                [tfidf_features.toarray(), numerical_features])

        logger.info(
            f"Feature combination completed in {time.time() - start:.2f} seconds")
        return combined

    def _pseudo_label_data(self, df):
        """Generate pseudo-labels for unlabeled data with three-cluster approach as per architecture diagram"""
        try:
            # Initialize attributes to store cluster models
            self.cluster_kmeans = None
            self.cluster_models = {}

            # If the dataframe is empty or already has labels, return as is
            if len(df) == 0 or 'label' in df.columns:
                return df

            # Train a simple classifier on the text features
            vectorizer = TfidfVectorizer(
                max_features=500, stop_words='english')
            X = vectorizer.fit_transform(df['text'])

            # Create initial labels based on linguistic features
            feature_df = pd.DataFrame()
            feature_df['word_count'] = df['text'].apply(
                lambda x: len(x.split()))
            feature_df['avg_word_length'] = df['text'].apply(lambda x: sum(
                len(word) for word in x.split()) / max(len(x.split()), 1))
            feature_df['unique_words_ratio'] = df['text'].apply(
                lambda x: len(set(x.split())) / max(len(x.split()), 1))
            feature_df['stopwords_ratio'] = df['text'].apply(lambda x: sum(
                1 for word in x.split() if word.lower() in STOPWORDS) / max(len(x.split()), 1))
            # Additional features for better cluster differentiation
            feature_df['sentence_count'] = df['text'].apply(
                lambda x: len([s for s in x.split('.') if s.strip()]))
            feature_df['avg_sentence_length'] = df['text'].apply(
                lambda x: np.mean([len(s.split()) for s in x.split('.') if s.strip()]) if len([s for s in x.split('.') if s.strip()]) > 0 else 0)
            feature_df['punctuation_ratio'] = df['text'].apply(
                lambda x: sum(1 for c in x if c in '.,!?;:') / len(x) if len(x) > 0 else 0)

            # Use more features for better clustering
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier
            from xgboost import XGBClassifier

            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(feature_df)

            # Combine with TF-IDF for better clustering
            combined_features = hstack([X, features_scaled])

            # Use K-means to find THREE natural clusters as shown in architecture diagram
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(combined_features)

            # Prepare cluster-specific data
            cluster_data = {}
            for cluster_id in range(3):
                cluster_mask = (clusters == cluster_id)
                cluster_data[cluster_id] = {
                    'mask': cluster_mask,
                    'features': combined_features[cluster_mask],
                    'df': df[cluster_mask].copy()
                }
                logger.info(
                    f"Cluster {cluster_id} size: {sum(cluster_mask)} samples")

            # Analyze cluster characteristics to determine potential labels
            cluster_features = pd.DataFrame()
            cluster_features['cluster'] = clusters
            cluster_features['word_count'] = feature_df['word_count'].values
            cluster_features['unique_ratio'] = feature_df['unique_words_ratio'].values
            cluster_features['avg_sentence_length'] = feature_df['avg_sentence_length'].values

            # Analyze cluster statistics
            cluster_stats = cluster_features.groupby('cluster').mean()
            logger.info(f"Cluster statistics:\n{cluster_stats}")

            # Determine characteristics of each cluster and assign initial labels
            # Lower unique word ratio and more extreme metrics likely indicate CG
            cluster_labels = {}

            # Sort clusters by unique_ratio (lower is more likely CG)
            sorted_clusters = cluster_stats.sort_values(
                'unique_ratio').index.tolist()

            # Most likely computer-generated cluster
            cg_cluster = sorted_clusters[0]
            # Most likely human-written cluster
            or_cluster = sorted_clusters[2]
            # Mixed/uncertain cluster
            mixed_cluster = sorted_clusters[1]

            # Initialize cluster label mapping
            cluster_labels = {
                cg_cluster: 'CG',     # Computer-generated
                or_cluster: 'OR',     # Original/human-written
                mixed_cluster: None   # Mixed/uncertain
            }

            logger.info(
                f"Identified clusters - CG: {cg_cluster}, OR: {or_cluster}, Mixed: {mixed_cluster}")

            # Train cluster-specific models as shown in architecture diagram
            cluster_models = {}

            # For each identifiable cluster (CG and OR), train a specialized model
            for cluster_id in [cg_cluster, or_cluster]:
                # Only use confidently labeled clusters
                if cluster_labels[cluster_id]:
                    # Create synthetic labels based on cluster identity
                    y_synthetic = np.ones(
                        sum(cluster_data[cluster_id]['mask']))
                    if cluster_labels[cluster_id] == 'OR':
                        y_synthetic = np.zeros(
                            sum(cluster_data[cluster_id]['mask']))

                    # Train specialized models for this cluster (as per architecture)
                    models = {
                        'logistic': LogisticRegression(max_iter=1000, random_state=42),
                        'rf': RandomForestClassifier(n_estimators=50, random_state=42),
                        'xgb': XGBClassifier(n_estimators=50, random_state=42)
                    }

                    # Train each model on this cluster's data
                    for name, model in models.items():
                        model.fit(cluster_data[cluster_id]
                                  ['features'], y_synthetic)

                    # Store trained models
                    cluster_models[cluster_id] = models
                    logger.info(
                        f"Trained specialized models for cluster {cluster_id} ({cluster_labels[cluster_id]})")

            # For mixed/uncertain cluster, use ensemble voting from the other clusters' models
            # This implements the "final evaluator" shown in the architecture diagram
            if sum(cluster_data[mixed_cluster]['mask']) > 0:
                mixed_features = cluster_data[mixed_cluster]['features']

                # Get predictions from both cluster models
                predictions = []
                for cluster_id, models in cluster_models.items():
                    cluster_pred = np.zeros(
                        sum(cluster_data[mixed_cluster]['mask']))
                    # Average predictions from all models in this cluster
                    for name, model in models.items():
                        if hasattr(model, 'predict_proba'):
                            cluster_pred += model.predict_proba(
                                mixed_features)[:, 1]
                        else:
                            cluster_pred += model.predict(mixed_features)
                    cluster_pred /= len(models)
                    predictions.append(cluster_pred)

                # Average predictions from all cluster models (ensemble/final evaluator)
                if predictions:
                    mixed_pred = np.mean(predictions, axis=0)
                    mixed_labels = (mixed_pred > 0.5).astype(int)
                    # Assign labels to mixed cluster
                    cluster_data[mixed_cluster]['df']['label'] = [
                        'CG' if p == 1 else 'OR' for p in mixed_labels]
                    logger.info(
                        f"Assigned labels to mixed cluster: {sum(mixed_labels)} CG, {sum(mixed_labels == 0)} OR")

            # Prepare the final labeled dataset
            # For CG and OR clusters, directly assign labels
            for cluster_id in [cg_cluster, or_cluster]:
                if cluster_labels[cluster_id]:
                    cluster_data[cluster_id]['df']['label'] = cluster_labels[cluster_id]

            # Combine all clusters
            df_with_labels = pd.concat([
                cluster_data[cg_cluster]['df'],
                cluster_data[or_cluster]['df'],
                cluster_data[mixed_cluster]['df']
            ], ignore_index=True)

            # Calculate confidence based on distance to cluster centers
            distances = kmeans.transform(combined_features)
            confidence = 1 - (np.min(distances, axis=1) /
                              np.sum(distances, axis=1))

            # Only use high confidence predictions
            high_conf_mask = confidence > 0.6  # Higher values mean higher confidence

            # Store the trained K-means model for future use
            self.cluster_kmeans = kmeans
            self.cluster_models = cluster_models

            logger.info(
                f"Generated {sum(high_conf_mask)} high-confidence pseudo-labels out of {len(df)} samples")

            # Return only high confidence samples
            return df_with_labels[high_conf_mask]

        except Exception as e:
            logger.error(f"Error in pseudo-labeling: {str(e)}")
            # Make sure attributes are cleared in case of error
            self.cluster_kmeans = None
            self.cluster_models = {}
            # Return empty dataframe on error
            return pd.DataFrame(columns=df.columns)

    def extract_linguistic_features(self, text):
        """Extract linguistic features for analysis"""
        cleaned_text = self._clean_text(text)
        words = cleaned_text.split()
        sentences = [s.strip() for s in cleaned_text.split('.') if s.strip()]

        features = {
            'length': len(cleaned_text),
            'word_count': len(words),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'unique_words_ratio': len(set(words)) / len(words) if words else 0,
            'punctuation_count': sum(1 for char in cleaned_text if char in '.,!?'),
            'sentence_count': max(len(sentences), 1),
            'caps_ratio': sum(1 for c in cleaned_text if c.isupper()) / len(cleaned_text) if len(cleaned_text) > 0 else 0,
            'stopwords_ratio': sum(1 for word in words if word.lower() in STOPWORDS) / len(words) if words else 0,
            'avg_sentence_length': sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0,
            'repetition_score': self._calculate_repetition(cleaned_text),
            'grammar_complexity': self._grammar_complexity(cleaned_text)
        }

        return features

    def _calculate_repetition(self, text):
        """Calculate text repetition score with validation"""
        try:
            words = word_tokenize(text.lower())
            if not words:
                return 0
            word_freq = pd.Series(words).value_counts()
            return (word_freq > 2).sum() / len(words)
        except Exception:
            return 0

    def _grammar_complexity(self, text):
        """Estimate grammatical complexity with error handling"""
        try:
            complex_markers = ['although', 'however',
                               'nevertheless', 'therefore', 'thus', 'moreover']
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if not sentences:
                return 0
            return sum(any(marker in s.lower() for marker in complex_markers)
                       for s in sentences) / len(sentences)
        except Exception:
            return 0

    def _augment_minority_class(self, df):
        """Augment the minority class to help with class imbalance"""
        # Count classes
        class_counts = df['label'].value_counts()
        majority_class = class_counts.idxmax()
        minority_class = class_counts.idxmin()

        logger.info(
            f"Class distribution before augmentation: {class_counts.to_dict()}")

        # Extract minority class samples
        minority_samples = df[df['label'] == minority_class]
        majority_samples = df[df['label'] == majority_class]

        # Determine how many samples to generate
        n_to_add = len(majority_samples) - len(minority_samples)

        if n_to_add <= 0:
            logger.info("No augmentation needed, classes are balanced")
            return df

        logger.info(
            f"Augmenting {n_to_add} samples for class '{minority_class}'")

        # Simple augmentation techniques
        augmented_samples = []

        # Sample with replacement
        indices = np.random.choice(
            len(minority_samples), size=n_to_add, replace=True)

        for idx in indices:
            sample = minority_samples.iloc[idx].copy()
            text = sample['text']

            # Apply one of several augmentation techniques
            aug_type = np.random.choice(['synonym', 'shuffle', 'delete'])

            if aug_type == 'synonym':
                # Replace some words with similar words
                words = text.split()
                for i in range(min(3, len(words))):
                    pos = np.random.randint(0, len(words))
                    words[pos] = words[pos] + \
                        's' if not words[pos].endswith(
                            's') else words[pos][:-1]
                sample['text'] = ' '.join(words)

            elif aug_type == 'shuffle':
                # Shuffle some sentences
                sentences = text.split('.')
                if len(sentences) > 1:
                    np.random.shuffle(sentences)
                    sample['text'] = '.'.join(sentences)

            elif aug_type == 'delete':
                # Delete some words
                words = text.split()
                if len(words) > 10:
                    to_delete = np.random.choice(
                        len(words), size=min(3, len(words)//10), replace=False)
                    sample['text'] = ' '.join(
                        [w for i, w in enumerate(words) if i not in to_delete])

            # Re-compute features for augmented samples
            augmented_samples.append(sample)

        # Create DataFrame from augmented samples and combine with original data
        augmented_df = pd.DataFrame(augmented_samples)
        result_df = pd.concat([df, augmented_df], ignore_index=True)

        logger.info(
            f"Class distribution after augmentation: {result_df['label'].value_counts().to_dict()}")
        return result_df

    def preprocess_dataset(self, dataset_path):
        """Process a dataset that already has labeled and unlabeled data together"""
        try:
            self.start_time = time.time()
            logger.info(f"Starting preprocessing of {dataset_path}")

            # Load dataset
            df = pd.read_csv(dataset_path)
            logger.info(f"Loaded {len(df)} rows from dataset")

            # Determine the column containing review text
            text_column = None
            for col in ['review_text', 'text', 'review']:
                if col in df.columns:
                    text_column = col
                    break

            if text_column is None:
                error_msg = f"Could not find text column in dataset. Available columns: {df.columns.tolist()}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.info(f"Using column '{text_column}' for review text")

            # Copy text to a standardized column name for further processing
            df['text'] = df[text_column]

            # Apply preprocessing
            df = self._process_dataset(df)
            logger.info(
                f"After preprocessing: {len(df)} samples remaining")

            # Check for labeled and unlabeled data
            if 'label' in df.columns:
                labeled_df = df[df['label'].notna()].copy()
                unlabeled_df = df[df['label'].isna()].copy()
                logger.info(
                    f"Found {len(labeled_df)} labeled and {len(unlabeled_df)} unlabeled samples")

                # Apply pseudo-labeling to unlabeled data if it exists
                if len(unlabeled_df) > 0:
                    logger.info(
                        "Applying cluster-based pseudo-labeling to unlabeled data...")
                    pseudo_labeled_df = self._pseudo_label_data(unlabeled_df)
                    logger.info(
                        f"Generated {len(pseudo_labeled_df)} pseudo-labeled samples")

                    # Combine with labeled data if pseudo-labels were generated
                    if len(pseudo_labeled_df) > 0:
                        combined_df = pd.concat(
                            [labeled_df, pseudo_labeled_df], ignore_index=True)
                        logger.info(
                            f"Combined dataset now has {len(combined_df)} samples")
                    else:
                        combined_df = labeled_df
                        logger.info(
                            "No reliable pseudo-labels generated, using only labeled data")
                else:
                    combined_df = labeled_df
            else:
                logger.warning("No 'label' column found in dataset")
                logger.info(
                    "Applying clustering and pseudo-labeling to all data...")
                combined_df = self._pseudo_label_data(df)
                logger.info(f"Generated {len(combined_df)} labeled samples")

            # Apply class balancing
            class_counts = combined_df['label'].value_counts()
            logger.info(f"Class distribution: {class_counts.to_dict()}")

            majority_class = class_counts.idxmax()
            minority_class = class_counts.idxmin()
            if class_counts[majority_class] > 2 * class_counts[minority_class]:
                logger.info("Applying class balancing...")
                majority_samples = combined_df[combined_df['label']
                                               == majority_class]
                minority_samples = combined_df[combined_df['label']
                                               == minority_class]

                # Downsample majority class to 2x minority
                balanced_majority = majority_samples.sample(
                    n=min(len(minority_samples) * 2, len(majority_samples)),
                    random_state=42
                )

                combined_df = pd.concat(
                    [balanced_majority, minority_samples], ignore_index=True)
                logger.info(
                    f"After balancing: {len(combined_df)} samples, distribution: {combined_df['label'].value_counts().to_dict()}")

            # Split into training and validation sets
            train_df, val_df = train_test_split(
                combined_df,
                test_size=0.2,
                random_state=42,
                stratify=combined_df['label']
            )
            logger.info(
                f"Split into {len(train_df)} training and {len(val_df)} validation samples")

            # Convert labels to integers for training
            train_df['label_int'] = (train_df['label'] == 'CG').astype(int)
            val_df['label_int'] = (val_df['label'] == 'CG').astype(int)

            # Generate TF-IDF features
            logger.info("Generating TF-IDF features...")
            start_tfidf = time.time()
            X_train_tfidf = self.vectorizer.fit_transform(train_df['text'])
            X_val_tfidf = self.vectorizer.transform(val_df['text'])
            logger.info(
                f"TF-IDF vectorization completed in {time.time() - start_tfidf:.2f} seconds")

            # Get the number of TF-IDF features
            n_tfidf_features = X_train_tfidf.shape[1]
            logger.info(f"Number of TF-IDF features: {n_tfidf_features}")

            numerical_features = [
                'length', 'word_count', 'avg_word_length', 'unique_words_ratio',
                'punctuation_count', 'sentence_count', 'caps_ratio', 'stopwords_ratio',
                'avg_sentence_length', 'repetition_score', 'grammar_complexity'
            ]

            logger.info("Scaling numerical features...")
            start_scaling = time.time()
            X_train_num = self.scaler.fit_transform(
                train_df[numerical_features])
            X_val_num = self.scaler.transform(val_df[numerical_features])
            logger.info(
                f"Feature scaling completed in {time.time() - start_scaling:.2f} seconds")

            # Calculate total features
            total_features = n_tfidf_features + len(numerical_features)
            logger.info(f"Total number of features: {total_features}")

            # Combine features
            logger.info("Combining features...")
            X_train = self._combine_features(X_train_tfidf, X_train_num)
            X_val = self._combine_features(X_val_tfidf, X_val_num)

            y_train = train_df['label_int'].values
            y_val = val_df['label_int'].values

            logger.info(
                f"All preprocessing completed in {time.time() - self.start_time:.2f} seconds")

            # Store information about any cluster models we created during preprocessing
            # This allows leveraging them later for prediction
            if hasattr(self, 'cluster_kmeans') and hasattr(self, 'cluster_models'):
                logger.info("Using cluster-specific models from preprocessing")
                # Save models for later use in prediction
                os.makedirs('models/clusters', exist_ok=True)
                joblib.dump(self.cluster_kmeans,
                            'models/clusters/kmeans.joblib')
                for cluster_id, models in self.cluster_models.items():
                    for name, model in models.items():
                        joblib.dump(
                            model, f'models/clusters/cluster_{cluster_id}_{name}_model.joblib')
                logger.info("Saved cluster-specific models")

            return X_train, X_val, y_train, y_val

        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def _augment_minority_class_reduced(self, df):
        """Reduced augmentation strategy to avoid reinforcing patterns"""
        # Count classes
        class_counts = df['label'].value_counts()
        majority_class = class_counts.idxmax()
        minority_class = class_counts.idxmin()

        logger.info(
            f"Class distribution before augmentation: {class_counts.to_dict()}")

        # Extract minority class samples
        minority_samples = df[df['label'] == minority_class]
        majority_samples = df[df['label'] == majority_class]

        # Determine how many samples to generate - use at most 50% augmentation
        n_to_add = min(len(majority_samples) - len(minority_samples),
                       int(len(minority_samples) * 0.5))

        if n_to_add <= 0:
            logger.info("No augmentation needed, classes are balanced")
            return df

        logger.info(
            f"Augmenting {n_to_add} samples for class '{minority_class}' (reduced strategy)")

        # Simple augmentation technique - only use deletion which is less likely to create artificial patterns
        augmented_samples = []

        # Sample without replacement where possible to avoid duplicates
        replace = len(minority_samples) < n_to_add
        indices = np.random.choice(
            len(minority_samples), size=n_to_add, replace=replace)

        for idx in indices:
            sample = minority_samples.iloc[idx].copy()
            text = sample['text']

            # Only use word deletion, which is less likely to create artificial patterns
            words = text.split()
            if len(words) > 10:
                # Delete fewer words
                to_delete = np.random.choice(
                    len(words), size=min(2, len(words)//15), replace=False)
                sample['text'] = ' '.join(
                    [w for i, w in enumerate(words) if i not in to_delete])
                augmented_samples.append(sample)

        # Create DataFrame from augmented samples and combine with original data
        if augmented_samples:
            augmented_df = pd.DataFrame(augmented_samples)
            result_df = pd.concat([df, augmented_df], ignore_index=True)
        else:
            result_df = df

        logger.info(
            f"Class distribution after augmentation: {result_df['label'].value_counts().to_dict()}")
        return result_df


class ReviewClassifier:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['model']
        self.model = None
        self.xgb_model = None
        self.ensemble_model = None
        self.use_ensemble = True  # Enable the ensemble approach
        # Use consistent feature count
        self.preprocessor = DataPreprocessor(max_features=5000)

    def build_model(self, input_shape):
        # Neural Network Model with balanced regularization
        inputs = tf.keras.Input(shape=(input_shape,))

        # Reduce L2 regularization
        regularizer = tf.keras.regularizers.l2(
            0.008)  # Decreased from 0.015 to 0.008

        # Increase model capacity but keep reasonable regularization
        # Restored from 96→48 to 128→64 with moderate dropout
        x = Dense(128, activation='relu',
                  kernel_regularizer=regularizer)(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)  # Decreased dropout from 0.6 to 0.4

        x = Dense(64, activation='relu', kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)  # Decreased dropout from 0.6 to 0.4

        # Output layer (1 neuron for binary classification)
        outputs = Dense(1, activation='sigmoid',
                        kernel_regularizer=regularizer)(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Compile model with balanced learning rate and weight decay
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0002,  # Increased from 0.0001
            weight_decay=8e-5  # Decreased from 1.5e-4
        )

        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(),
                     tf.keras.metrics.Recall(), 'AUC']
        )
        self.model = model

        # Update XGBoost model with more balanced regularization
        self.xgb_model = xgb.XGBClassifier(
            learning_rate=0.03,  # Increased from 0.02
            n_estimators=120,
            max_depth=4,         # Increased from 3 to provide more capacity
            min_child_weight=3,  # Decreased from 4
            gamma=0.2,           # Decreased from 0.3
            subsample=0.7,
            colsample_bytree=0.7,  # Slightly increased from 0.65
            objective='binary:logistic',
            scale_pos_weight=1.2,  # Added weight to focus more on positive class
            tree_method='hist',
            reg_alpha=0.1,       # Decreased from 0.15
            reg_lambda=1.0,      # Decreased from 1.2
            eval_metric='auc',
            early_stopping_rounds=10  # Decreased from 12
        )

        return model

    def train(self, X_train, y_train, X_val=None, y_val=None):
        # Build model with correct input shape
        if self.model is None:
            input_shape = X_train.shape[1]
            logger.info(f"Building model with input shape: {input_shape}")
            self.model = self.build_model(input_shape)

        # Calculate class weights to handle imbalanced data
        class_weights = None
        if self.config['use_class_weights']:
            # Use a more balanced approach to class weights that puts slightly
            # more emphasis on the minority class but less extreme than before
            class_count = np.bincount(y_train)
            total = len(y_train)
            n_classes = len(class_count)

            # Adjust weight calculation to be more balanced
            class_weights = {}
            for i in range(n_classes):
                # This is a more gentle weighting than the default
                class_weights[i] = total / (n_classes * class_count[i]) * 0.8

                # For the minority class, give it a slightly higher weight
                if class_count[i] == min(class_count):
                    class_weights[i] *= 1.2

            logger.info(f"Using balanced class weights: {class_weights}")

        # Create custom learning rate scheduler with gentler decay
        def lr_scheduler(epoch, lr):
            if epoch < 2:  # Short warm-up phase
                return 0.0002
            elif epoch < 10:
                return 0.0002  # Hold steady for early epochs
            elif epoch < 15:
                return 0.00015  # First gentle reduction
            elif epoch < 20:
                return 0.0001  # Second gentle reduction
            else:
                return 0.00005  # Final learning rate, higher than before

        # Train neural network with adjusted early stopping
        nn_history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            batch_size=32,
            epochs=30,
            class_weight=class_weights,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=7,        # Increased from 5 to allow more training time
                    restore_best_weights=True,
                    min_delta=0.001   # Increased from 0.0005 to prevent stopping too early
                ),
                tf.keras.callbacks.LearningRateScheduler(lr_scheduler),
                tf.keras.callbacks.TensorBoard(
                    log_dir='./logs',
                    histogram_freq=1,
                    write_graph=True,
                    write_images=True,
                    update_freq='epoch',
                    profile_batch=0
                )
            ]
        )

        # Train XGBoost model with early stopping
        logger.info("Training XGBoost model...")
        eval_set = [(X_val, y_val)] if X_val is not None else None

        self.xgb_model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=True
        )

        # Create and train a Random Forest with balanced regularization
        rf_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,            # Increased from 6 to allow more complexity
            min_samples_split=15,   # Decreased from 20
            min_samples_leaf=5,     # Decreased from 8
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,
            n_jobs=-1,
            # Changed from 'balanced' to balanced_subsample
            class_weight='balanced_subsample',
            max_samples=0.8,        # Increased from 0.7 to use more samples
            random_state=42
        )
        rf_model.fit(X_train, y_train)

        # Save individual models
        self.rf_model = rf_model

        logger.info("All models trained successfully")

        return nn_history

    def evaluate(self, X_val, y_val):
        # Get predictions from neural network
        nn_pred_proba = self.model.predict(X_val)
        nn_pred = (nn_pred_proba > 0.5).astype(int)

        # Get predictions from XGBoost
        xgb_pred_proba = self.xgb_model.predict_proba(X_val)[:, 1]
        xgb_pred = (xgb_pred_proba > 0.5).astype(int)

        # Get predictions from Random Forest
        rf_pred_proba = self.rf_model.predict_proba(X_val)[:, 1]
        rf_pred = (rf_pred_proba > 0.5).astype(int)

        # Calculate metrics for each model
        from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, balanced_accuracy_score

        nn_auc = roc_auc_score(y_val, nn_pred_proba)
        xgb_auc = roc_auc_score(y_val, xgb_pred_proba)
        rf_auc = roc_auc_score(y_val, rf_pred_proba)

        # Also consider F1 scores
        nn_f1 = f1_score(y_val, nn_pred)
        xgb_f1 = f1_score(y_val, xgb_pred)
        rf_f1 = f1_score(y_val, rf_pred)

        # Calculate balanced accuracy
        nn_bal_acc = balanced_accuracy_score(y_val, nn_pred)
        xgb_bal_acc = balanced_accuracy_score(y_val, xgb_pred)
        rf_bal_acc = balanced_accuracy_score(y_val, rf_pred)

        # Combined score with balanced emphasis on different metrics
        nn_score = 0.5 * nn_auc + 0.3 * nn_f1 + 0.2 * nn_bal_acc
        xgb_score = 0.5 * xgb_auc + 0.3 * xgb_f1 + 0.2 * xgb_bal_acc
        rf_score = 0.5 * rf_auc + 0.3 * rf_f1 + 0.2 * rf_bal_acc

        # Calculate dynamically based on model performance but with limits
        total_score = nn_score + xgb_score + rf_score
        nn_weight = min(max(0.3, nn_score / total_score), 0.4)
        xgb_weight = min(max(0.35, xgb_score / total_score), 0.45)

        # Ensure weights sum to 1
        rf_weight = 1.0 - nn_weight - xgb_weight

        # Log detailed metrics
        logger.info(
            f"Model AUC scores: NN={nn_auc:.4f}, XGB={xgb_auc:.4f}, RF={rf_auc:.4f}")
        logger.info(
            f"Model F1 scores: NN={nn_f1:.4f}, XGB={xgb_f1:.4f}, RF={rf_f1:.4f}")
        logger.info(
            f"Model balanced accuracy: NN={nn_bal_acc:.4f}, XGB={xgb_bal_acc:.4f}, RF={rf_bal_acc:.4f}")
        logger.info(
            f"Using ensemble weights: NN={nn_weight:.2f}, XGB={xgb_weight:.2f}, RF={rf_weight:.2f}")

        # Ensemble predictions with balanced weights
        ensemble_pred_proba = (nn_weight * nn_pred_proba.reshape(-1) +
                               xgb_weight * xgb_pred_proba +
                               rf_weight * rf_pred_proba)

        # Feature selection based on importance
        try:
            # Get feature importances from XGBoost
            importances = self.xgb_model.feature_importances_

            # Log top and bottom features
            if hasattr(self, 'feature_names') and len(self.feature_names) == len(importances):
                # Sort features by importance
                indices = np.argsort(importances)[::-1]

                # Log top 20 features
                logger.info("Top 20 most important features:")
                for i in range(min(20, len(indices))):
                    logger.info(
                        f"{self.feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

                # Log least important features
                logger.info("10 least important features:")
                for i in range(1, min(11, len(indices))+1):
                    idx = indices[-i]
                    logger.info(
                        f"{self.feature_names[idx]}: {importances[idx]:.4f}")
        except Exception as e:
            logger.warning(f"Could not analyze feature importance: {str(e)}")

        # Balanced threshold finding approach
        from sklearn.metrics import precision_recall_curve, roc_curve, f1_score

        # Get confusion matrices at various thresholds to better understand tradeoffs
        thresholds_to_analyze = [0.3, 0.4, 0.5, 0.6, 0.7]
        for threshold in thresholds_to_analyze:
            preds = (ensemble_pred_proba > threshold).astype(int)
            cm = confusion_matrix(y_val, preds)
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / \
                (precision + recall) if (precision + recall) > 0 else 0
            logger.info(f"Threshold {threshold:.2f}: TP={tp}, FP={fp}, TN={tn}, FN={fn}, "
                        f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

        # Find threshold that maximizes F1 score with a grid search
        best_f1 = 0
        best_f1_threshold = 0.5
        for threshold in np.arange(0.35, 0.65, 0.01):
            preds = (ensemble_pred_proba > threshold).astype(int)
            f1 = f1_score(y_val, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_f1_threshold = threshold

        logger.info(
            f"Best F1 score {best_f1:.4f} at threshold {best_f1_threshold:.3f}")

        # Find threshold that gives balanced accuracy
        fpr, tpr, roc_thresholds = roc_curve(y_val, ensemble_pred_proba)
        optimal_idx = np.argmax(tpr - fpr)
        balanced_threshold = roc_thresholds[optimal_idx]
        logger.info(f"Balanced accuracy threshold: {balanced_threshold:.3f}")

        # Find threshold with ideal precision-recall tradeoff (prioritize recall more)
        precision, recall, pr_thresholds = precision_recall_curve(
            y_val, ensemble_pred_proba)

        # Compute F2 score (emphasizes recall more than precision)
        f2_scores = []
        # -1 because there's one more precision/recall value than thresholds
        for i in range(len(precision)-1):
            if i < len(pr_thresholds):  # Ensure we don't go out of bounds
                p, r = precision[i], recall[i]
                if p > 0 and r > 0:
                    # F2 score formula: (1+beta^2) * (precision*recall) / (beta^2*precision + recall)
                    # where beta=2 to emphasize recall
                    f2 = 5 * p * r / (4 * p + r)
                    f2_scores.append((pr_thresholds[i], f2))

        # Get best F2 threshold if we have scores
        if f2_scores:
            pr_threshold = max(f2_scores, key=lambda x: x[1])[0]
            logger.info(
                f"Best precision-recall tradeoff (F2) threshold: {pr_threshold:.3f}")
        else:
            pr_threshold = 0.5
            logger.info(
                f"Using default precision-recall threshold: {pr_threshold:.3f}")

        # Use a weighted combination of thresholds
        optimal_threshold = 0.45 * best_f1_threshold + \
            0.35 * balanced_threshold + 0.2 * pr_threshold
        logger.info(f"Selected optimal threshold: {optimal_threshold:.3f}")

        # Use optimal threshold for final predictions
        ensemble_pred = (ensemble_pred_proba > optimal_threshold).astype(int)

        # Evaluate individual models
        nn_metrics = {
            'accuracy': accuracy_score(y_val, nn_pred),
            'precision': precision_score(y_val, nn_pred),
            'recall': recall_score(y_val, nn_pred),
            'f1': f1_score(y_val, nn_pred),
            'auc': nn_auc,
            'balanced_acc': nn_bal_acc
        }

        xgb_metrics = {
            'accuracy': accuracy_score(y_val, xgb_pred),
            'precision': precision_score(y_val, xgb_pred),
            'recall': recall_score(y_val, xgb_pred),
            'f1': f1_score(y_val, xgb_pred),
            'auc': xgb_auc,
            'balanced_acc': xgb_bal_acc
        }

        rf_metrics = {
            'accuracy': accuracy_score(y_val, rf_pred),
            'precision': precision_score(y_val, rf_pred),
            'recall': recall_score(y_val, rf_pred),
            'f1': f1_score(y_val, rf_pred),
            'auc': rf_auc,
            'balanced_acc': rf_bal_acc
        }

        ensemble_metrics = {
            'accuracy': accuracy_score(y_val, ensemble_pred),
            'precision': precision_score(y_val, ensemble_pred),
            'recall': recall_score(y_val, ensemble_pred),
            'f1': f1_score(y_val, ensemble_pred),
            'auc': roc_auc_score(y_val, ensemble_pred_proba),
            'balanced_acc': balanced_accuracy_score(y_val, ensemble_pred),
            'threshold': optimal_threshold,
            'weights': {
                'nn': nn_weight,
                'xgb': xgb_weight,
                'rf': rf_weight
            }
        }

        # Log individual model performance
        logger.info(f"Neural Network metrics: {nn_metrics}")
        logger.info(f"XGBoost metrics: {xgb_metrics}")
        logger.info(f"Random Forest metrics: {rf_metrics}")
        logger.info(f"Ensemble metrics: {ensemble_metrics}")

        # Save the optimal threshold and weights
        self.optimal_threshold = optimal_threshold
        self.model_weights = {
            'nn': nn_weight,
            'xgb': xgb_weight,
            'rf': rf_weight
        }

        # Return ensemble metrics as the official evaluation
        return ensemble_metrics

    def predict(self, X):
        """Get predictions using the ensemble model with optimal weights, leveraging cluster-specific models if available"""
        if not hasattr(self, 'optimal_threshold'):
            self.optimal_threshold = 0.75  # Increased from 0.7 to reduce false positives
            self.model_weights = {'nn': 0.35, 'xgb': 0.45, 'rf': 0.2}

        # Check if cluster models are available
        use_cluster_models = False
        cluster_kmeans = None
        cluster_models = {}

        # Try to load cluster models if they exist
        try:
            if os.path.exists('models/clusters/kmeans.joblib'):
                cluster_kmeans = joblib.load('models/clusters/kmeans.joblib')

                # Find all cluster model files
                cluster_files = [f for f in os.listdir(
                    'models/clusters') if f.startswith('cluster_') and f.endswith('_model.joblib')]

                # Load each cluster model
                for file in cluster_files:
                    # Parse file name to get cluster ID and model type
                    parts = file.replace('_model.joblib', '').split('_')
                    cluster_id = int(parts[1])
                    model_type = parts[2]

                    # Initialize dictionary for this cluster if needed
                    if cluster_id not in cluster_models:
                        cluster_models[cluster_id] = {}

                    # Load the model
                    cluster_models[cluster_id][model_type] = joblib.load(
                        os.path.join('models/clusters', file))

                # Only use cluster models if we found both K-means and at least one model
                if cluster_models and len(cluster_models) > 0:
                    use_cluster_models = True
                    logger.info(
                        f"Using {len(cluster_models)} cluster-specific models for prediction")
        except Exception as e:
            logger.warning(f"Could not load cluster models: {str(e)}")
            use_cluster_models = False

        # If cluster models are available, use them
        if use_cluster_models and cluster_kmeans is not None:
            try:
                # Assign each sample to its cluster
                clusters = cluster_kmeans.predict(X)

                # Get predictions from each cluster's models
                cluster_predictions = np.zeros(len(X))

                # Track which samples were predicted by cluster models
                cluster_predicted = np.zeros(len(X), dtype=bool)

                # For each cluster
                for cluster_id, models in cluster_models.items():
                    # Get samples in this cluster
                    cluster_mask = (clusters == cluster_id)

                    # Skip if no samples in this cluster
                    if not np.any(cluster_mask):
                        continue

                    # Get cluster features
                    X_cluster = X[cluster_mask]

                    # Get predictions from each model in this cluster
                    model_predictions = np.zeros(
                        (sum(cluster_mask), len(models)))

                    # For each model in this cluster
                    for i, (name, model) in enumerate(models.items()):
                        # Get predictions
                        if hasattr(model, 'predict_proba'):
                            model_predictions[:, i] = model.predict_proba(X_cluster)[
                                :, 1]
                        else:
                            model_predictions[:, i] = model.predict(X_cluster)

                    # Average predictions from all models
                    avg_predictions = np.mean(model_predictions, axis=1)

                    # Store predictions
                    cluster_predictions[cluster_mask] = avg_predictions
                    cluster_predicted[cluster_mask] = True

                # For samples not assigned to any cluster with models, use standard models
                if not np.all(cluster_predicted):
                    # Get predictions from neural network
                    nn_pred_proba = self.model.predict(
                        X[~cluster_predicted]).reshape(-1)

                    # Get predictions from XGBoost
                    xgb_pred_proba = self.xgb_model.predict_proba(
                        X[~cluster_predicted])[:, 1]

                    # Get predictions from Random Forest
                    rf_pred_proba = self.rf_model.predict_proba(
                        X[~cluster_predicted])[:, 1]

                    # Apply optimal weights from validation
                    ensemble_pred_proba = (
                        self.model_weights['nn'] * nn_pred_proba +
                        self.model_weights['xgb'] * xgb_pred_proba +
                        self.model_weights['rf'] * rf_pred_proba
                    )

                    # Store predictions
                    cluster_predictions[~cluster_predicted] = ensemble_pred_proba

                # Apply threshold to get final predictions
                ensemble_pred = (cluster_predictions >
                                 self.optimal_threshold).astype(int)

                # Return both probability and class prediction
                return cluster_predictions, ensemble_pred

            except Exception as e:
                logger.warning(f"Error using cluster models: {str(e)}")
                # Fall back to standard prediction
                pass

        # Standard prediction (used if cluster models are not available or failed)
        # Get predictions from all models
        nn_pred_proba = self.model.predict(X).reshape(-1)
        xgb_pred_proba = self.xgb_model.predict_proba(X)[:, 1]
        rf_pred_proba = self.rf_model.predict_proba(X)[:, 1]

        # Apply optimal weights from validation
        ensemble_pred_proba = (
            self.model_weights['nn'] * nn_pred_proba +
            self.model_weights['xgb'] * xgb_pred_proba +
            self.model_weights['rf'] * rf_pred_proba
        )

        # Apply threshold when returning classification predictions
        # but not when returning probabilities for further analysis
        ensemble_pred = (ensemble_pred_proba >
                         self.optimal_threshold).astype(int)

        # Return both probability and class prediction
        return ensemble_pred_proba, ensemble_pred

    def analyze_review(self, review_text, vectorizer):
        try:
            # Clean and preprocess the review
            cleaned_text = self.preprocessor._clean_text(review_text)

            # Get TF-IDF features
            X_tfidf = vectorizer.transform([cleaned_text])

            # Get numerical features
            numerical_features = self.preprocessor.extract_linguistic_features(
                cleaned_text)
            numerical_array = np.array([list(numerical_features.values())])

            # Scale numerical features
            X_num = self.preprocessor.scaler.transform(numerical_array)

            # Combine features
            X = np.hstack([X_tfidf.toarray(), X_num])

            # Get prediction and confidence using ensemble
            pred_proba, pred = self.predict(X)
            # Get the first element since we only have one sample
            pred_proba = pred_proba[0]

            # Determine prediction and confidence
            prediction = 'Computer Generated' if pred[0] == 1 else 'Original'
            confidence = float(max(pred_proba, 1-pred_proba))

            # Get detailed analysis
            analysis = {
                'prediction': prediction,
                'confidence': confidence,
                'raw_probability': float(pred_proba),
                'threshold_used': float(self.optimal_threshold) if hasattr(self, 'optimal_threshold') else 0.75,
                'reasoning': self._get_reasoning(cleaned_text, confidence, prediction)
            }

            return analysis

        except Exception as e:
            logger.error(f"Error in analyzing review: {str(e)}")
            raise

    def _get_reasoning(self, text, confidence, prediction):
        """Generate reasoning for the prediction"""
        reasons = []

        # Avoid repeated calculations by checking prediction first
        if prediction == 'Computer Generated':
            if self._check_repetitive_phrases(text):
                reasons.append(
                    "Contains repetitive phrases typical of AI text")
            if self._check_generic_expressions(text):
                reasons.append(
                    "Uses generic expressions commonly found in AI text")
            if self._check_sentiment_consistency(text):
                reasons.append("Shows inconsistent sentiment patterns")
        else:
            if not self._check_natural_patterns(text):
                reasons.append(
                    "Shows natural language variation and authenticity")

        # Add confidence-based reasoning
        if confidence > 0.9:
            reasons.append(
                f"High confidence ({confidence:.2f}) in {prediction} classification")
        elif confidence < 0.6:
            reasons.append(
                f"Low confidence ({confidence:.2f}) suggests mixed signals")

        return reasons

    def _check_repetitive_phrases(self, text):
        """Check for repeated phrases - optimized with early return"""
        phrases = ["I love", "very good", "great product",
                   "highly recommend", "excellent quality"]

        # Use a more efficient approach - return as soon as we find a repetition
        text_lower = text.lower()
        for phrase in phrases:
            if text_lower.count(phrase) > 1:
                return True
        return False

    def _check_generic_expressions(self, text):
        """Check for generic expressions - optimized with early return"""
        generic_phrases = [
            "the only reason",
            "we have had",
            "I will keep",
            "very pretty",
            "the materials are good",
            "the quality is good"
        ]

        # Convert to lowercase once
        text_lower = text.lower()

        # Return as soon as we find a match
        for phrase in generic_phrases:
            if phrase in text_lower:
                return True
        return False

    def _check_sentiment_consistency(self, text):
        """Check for inconsistent sentiment - optimized"""
        # Convert to lowercase once
        text_lower = text.lower()

        positive = ["great", "good", "excellent", "amazing", "love"]
        negative = ["bad", "poor", "terrible", "hate", "awful"]

        # Check positives first (often more common)
        has_positive = any(word in text_lower for word in positive)

        # Early return if no positive found
        if not has_positive:
            return False

        # Now check negatives
        has_negative = any(word in text_lower for word in negative)

        return has_negative

    def _check_natural_patterns(self, text):
        """Check for natural language patterns - optimized with compiled regex"""
        # Compile regex patterns only once (module level would be better)
        if not hasattr(self, '_compiled_patterns'):
            self._compiled_patterns = [
                re.compile(r'\b(however|nevertheless|although)\b'),
                re.compile(r'\b(specifically|particularly|especially)\b'),
                re.compile(r'\b(honestly|truthfully|frankly)\b'),
                re.compile(r'[!?]{2,}'),
                re.compile(r'\b(love|hate|amazing|terrible)\b.*but.*')
            ]

        # Convert to lowercase once
        text_lower = text.lower()

        # Return early if any pattern matches
        for pattern in self._compiled_patterns:
            if pattern.search(text_lower):
                return False
        return True

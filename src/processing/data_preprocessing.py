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
        """Generate pseudo-labels for unlabeled data - optimized version"""
        start = time.time()

        # Vectorized operations where possible
        # Create a composite score based on multiple features
        df['cg_score'] = (
            (df['repetition_score'] > 0.3).astype(float) * 0.2 +
            (df['unique_words_ratio'] < 0.4).astype(float) * 0.2 +
            (df['grammar_complexity'] < 0.1).astype(float) * 0.2 +
            (df['stopwords_ratio'] > 0.5).astype(float) * 0.2 +
            (df['avg_sentence_length'] > 20).astype(float) * 0.2
        )

        # Apply labels based on confidence score
        df['label'] = np.where(df['cg_score'] >= 0.6, 'CG', 'OR')

        # Only keep medium-to-high confidence samples
        result_df = df[df['cg_score'] >= 0.4].copy()

        logger.info(
            f"Pseudo-labeling completed in {time.time() - start:.2f} seconds")
        return result_df

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
            logger.info("Loading and preprocessing dataset")

            # Load dataset with latin1 encoding
            try:
                dataset_df = pd.read_csv(dataset_path, encoding='latin1')
                logger.info(f"Dataset size: {len(dataset_df)}")

                # Validate required columns
                if 'review_text' not in dataset_df.columns:
                    if 'review' in dataset_df.columns:
                        dataset_df['review_text'] = dataset_df['review']
                        logger.info("Using 'review' column as 'review_text'")
                    else:
                        raise ValueError(
                            f"Missing required column 'review_text' or 'review' in {dataset_path}")

                # Ensure label column exists with better criteria
                if 'label' not in dataset_df.columns:
                    if 'review_sentiment' in dataset_df.columns:
                        # Use clearer thresholds for sentiment-based labeling
                        # This is still not ideal, but better than using median
                        dataset_df['review_sentiment'] = dataset_df['review_sentiment'].astype(
                            float)
                        # Use fixed thresholds instead of median
                        dataset_df['label'] = dataset_df['review_sentiment'].apply(
                            lambda x: 'CG' if x < 0.3 else (
                                'OR' if x > 0.7 else None)
                        )
                        # Remove uncertain labels (those between 0.3 and 0.7)
                        dataset_df = dataset_df[dataset_df['label'].notna()].copy(
                        )
                        logger.info(
                            f"Created 'label' column using review_sentiment with fixed thresholds. Remaining samples: {len(dataset_df)}")
                    elif 'ratings' in dataset_df.columns:
                        # Use a more nuanced approach for ratings
                        dataset_df['ratings'] = pd.to_numeric(
                            dataset_df['ratings'], errors='coerce')
                        # Only use very low (1-2) and very high (4-5) ratings
                        dataset_df['label'] = dataset_df['ratings'].apply(
                            lambda x: 'CG' if x <= 2 else (
                                'OR' if x >= 4 else None)
                        )
                        # Remove uncertain labels (3 star ratings)
                        dataset_df = dataset_df[dataset_df['label'].notna()].copy(
                        )
                        logger.info(
                            f"Created 'label' column using ratings. Remaining samples: {len(dataset_df)}")
                    else:
                        logger.error(
                            "No appropriate column found for labels. Cannot proceed without labels.")
                        raise ValueError(
                            "Cannot create labels from available data")

            except Exception as e:
                logger.error(f"Error loading dataset: {str(e)}")
                raise

            # Process dataset with progress logging
            logger.info("Processing dataset...")
            dataset_df = self._process_dataset(dataset_df)
            logger.info(
                f"Dataset processing completed in {time.time() - self.start_time:.2f} seconds")
            logger.info(
                f"After processing - dataset size: {len(dataset_df)}")

            # Data augmentation for the minority class - REDUCED to prevent reinforcing patterns
            logger.info(
                "Performing limited data augmentation for class balance...")
            dataset_df = self._augment_minority_class_reduced(dataset_df)
            logger.info(
                f"After augmentation - dataset size: {len(dataset_df)}")

            # Split into training and validation with stratification
            logger.info("Splitting data into train and validation sets...")
            train_df, val_df = train_test_split(
                dataset_df,
                test_size=0.2,
                random_state=42,
                stratify=dataset_df['label']
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
        # Neural Network Model with much stronger regularization
        inputs = tf.keras.Input(shape=(input_shape,))

        # Add stronger L2 regularization
        regularizer = tf.keras.regularizers.l2(
            0.01)  # Increased from 0.001 to 0.01

        # Much simpler architecture to prevent overfitting
        # Reduced from 512→256→128 to 128→64
        x = Dense(128, activation='relu',
                  kernel_regularizer=regularizer)(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)  # Increased dropout from 0.4 to 0.5

        x = Dense(64, activation='relu', kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)  # Increased dropout

        # Output layer (1 neuron for binary classification)
        outputs = Dense(1, activation='sigmoid',
                        kernel_regularizer=regularizer)(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Compile model with lower learning rate and stronger weight decay
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0002,  # Further reduced from 0.0005
            weight_decay=1e-4  # Increased from 1e-5
        )

        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(),
                     tf.keras.metrics.Recall(), 'AUC']
        )
        self.model = model

        # Update XGBoost model for better regularization
        self.xgb_model = xgb.XGBClassifier(
            learning_rate=0.03,  # Further reduced from 0.05
            n_estimators=100,
            max_depth=4,         # Reduced from 6 to prevent overfitting
            min_child_weight=3,  # Increased from 2
            gamma=0.2,           # Increased from 0.1
            subsample=0.6,       # Reduced from 0.7
            colsample_bytree=0.6,  # Reduced from 0.7
            objective='binary:logistic',
            scale_pos_weight=1,
            tree_method='hist',
            reg_alpha=0.1,       # Added L1 regularization
            reg_lambda=1.0,      # Added L2 regularization
            eval_metric='auc',
            early_stopping_rounds=10  # Add early stopping as a model parameter
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
            n_samples = len(y_train)
            n_classes = len(np.unique(y_train))

            class_count = np.bincount(y_train)
            class_weights = {i: n_samples / (n_classes * count)
                             for i, count in enumerate(class_count)}
            logger.info(f"Using class weights: {class_weights}")

        # Train neural network with stronger early stopping
        nn_history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            batch_size=32,
            epochs=30,      # Reduced from 50 to prevent overfitting
            class_weight=class_weights,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    # More aggressive early stopping (reduced from 5)
                    patience=3,
                    restore_best_weights=True,
                    min_delta=0.001
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.1,    # More aggressive reduction (0.2 → 0.1)
                    patience=2,    # Reduced patience from 3 to 2
                    min_lr=0.00001
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

        # Create and train a Random Forest with stronger regularization
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,          # Further reduced from 10
            min_samples_split=15,  # Increased from 10
            min_samples_leaf=5,    # Increased from 4
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,
            n_jobs=-1,
            class_weight='balanced',
            max_samples=0.7       # Use bagging with 70% of samples
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

        # Calculate optimal weights based on validation performance
        nn_f1 = f1_score(y_val, nn_pred)
        xgb_f1 = f1_score(y_val, xgb_pred)
        rf_f1 = f1_score(y_val, rf_pred)

        # Normalize to get weights
        total_f1 = nn_f1 + xgb_f1 + rf_f1
        nn_weight = nn_f1 / total_f1
        xgb_weight = xgb_f1 / total_f1
        rf_weight = rf_f1 / total_f1

        logger.info(
            f"Ensemble weights based on F1: NN={nn_weight:.2f}, XGB={xgb_weight:.2f}, RF={rf_weight:.2f}")

        # Ensemble predictions with calculated weights
        ensemble_pred_proba = (nn_weight * nn_pred_proba.reshape(-1) +
                               xgb_weight * xgb_pred_proba +
                               rf_weight * rf_pred_proba)
        ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)

        # Find optimal threshold on validation set that prioritizes reducing false positives
        from sklearn.metrics import roc_curve, precision_recall_curve

        # Get various threshold options
        fpr, tpr, roc_thresholds = roc_curve(y_val, ensemble_pred_proba)
        precision, recall, pr_thresholds = precision_recall_curve(
            y_val, ensemble_pred_proba)

        # Find threshold that maximizes TPR - FPR (balanced accuracy)
        optimal_idx = np.argmax(tpr - fpr)
        roc_threshold = roc_thresholds[optimal_idx]

        # Find threshold that gives at least 0.95 precision (to reduce false positives)
        # Note: precision_recall_curve returns precision and recall of length n_thresholds + 1
        # The thresholds correspond to the precision and recall values at indices 1 to n_thresholds + 1
        try:
            # Find indices where precision is at least 0.95
            high_precision_idx = np.where(precision[:-1] >= 0.95)[0]

            # If we found some high precision points, use the threshold that gives highest recall
            if len(high_precision_idx) > 0:
                # Get the index with the highest recall among high-precision points
                best_idx = high_precision_idx[np.argmax(
                    recall[high_precision_idx])]
                pr_threshold = pr_thresholds[best_idx]
            else:
                # If no threshold gives 0.95 precision, use a high default
                pr_threshold = 0.7
        except Exception as e:
            logger.warning(
                f"Could not compute high precision threshold: {str(e)}. Using default 0.7")
            pr_threshold = 0.7

        # Use the higher of the two thresholds to be more conservative about CG predictions
        optimal_threshold = max(roc_threshold, pr_threshold)
        logger.info(f"Balanced accuracy threshold: {roc_threshold:.3f}")
        logger.info(f"High precision threshold: {pr_threshold:.3f}")
        logger.info(f"Selected optimal threshold: {optimal_threshold:.3f}")

        # Use optimal threshold for final predictions
        ensemble_pred = (ensemble_pred_proba > optimal_threshold).astype(int)

        # Evaluate individual models
        nn_metrics = {
            'accuracy': accuracy_score(y_val, nn_pred),
            'precision': precision_score(y_val, nn_pred),
            'recall': recall_score(y_val, nn_pred),
            'f1': f1_score(y_val, nn_pred)
        }

        xgb_metrics = {
            'accuracy': accuracy_score(y_val, xgb_pred),
            'precision': precision_score(y_val, xgb_pred),
            'recall': recall_score(y_val, xgb_pred),
            'f1': f1_score(y_val, xgb_pred)
        }

        rf_metrics = {
            'accuracy': accuracy_score(y_val, rf_pred),
            'precision': precision_score(y_val, rf_pred),
            'recall': recall_score(y_val, rf_pred),
            'f1': f1_score(y_val, rf_pred)
        }

        ensemble_metrics = {
            'accuracy': accuracy_score(y_val, ensemble_pred),
            'precision': precision_score(y_val, ensemble_pred),
            'recall': recall_score(y_val, ensemble_pred),
            'f1': f1_score(y_val, ensemble_pred),
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
        """Get predictions using the ensemble model with optimal weights"""
        if not hasattr(self, 'optimal_threshold'):
            self.optimal_threshold = 0.7  # Increased from 0.5 to reduce false positives
            self.model_weights = {'nn': 0.4, 'xgb': 0.4, 'rf': 0.2}

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

        return ensemble_pred_proba

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
            pred_proba = self.predict(X)[0]

            # Determine prediction and confidence
            prediction = 'Computer Generated' if pred_proba > 0.5 else 'Original'
            confidence = float(max(pred_proba, 1 - pred_proba))

            # Get detailed analysis
            analysis = {
                'prediction': prediction,
                'confidence': confidence,
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

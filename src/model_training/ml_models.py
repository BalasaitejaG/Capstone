import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
import seaborn as sns
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MLModels:
    """Class to implement and train additional ML models"""

    def __init__(self):
        """Initialize the models"""
        self.models = {}
        self.init_models()

    def init_models(self):
        """Initialize all ML models with optimized hyperparameters"""
        # Support Vector Machine
        self.models['svm'] = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=42
        )

        # K-Nearest Neighbors
        self.models['knn'] = KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            algorithm='auto',
            leaf_size=30,
            p=2,  # Euclidean distance
            n_jobs=-1
        )

        # Logistic Regression
        self.models['logistic'] = LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='liblinear',
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )

        # Naive Bayes
        self.models['naive_bayes'] = GaussianNB()

    def train_models(self, X_train, y_train, X_val=None, y_val=None):
        """Train all models and evaluate on validation set if provided"""
        results = {}

        for name, model in self.models.items():
            logger.info(f"Training {name} model...")
            model.fit(X_train, y_train)

            # Save the trained model
            os.makedirs('models', exist_ok=True)
            joblib.dump(model, f'models/{name}_model.joblib')

            # Evaluate on validation set if provided
            if X_val is not None and y_val is not None:
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val)[:, 1]

                accuracy = accuracy_score(y_val, y_pred)
                precision = precision_score(y_val, y_pred)
                recall = recall_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred)

                # Calculate ROC AUC
                fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
                roc_auc = auc(fpr, tpr)

                results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': roc_auc,
                    'fpr': fpr,
                    'tpr': tpr
                }

                logger.info(f"{name} model evaluation:")
                logger.info(f"  Accuracy: {accuracy:.4f}")
                logger.info(f"  Precision: {precision:.4f}")
                logger.info(f"  Recall: {recall:.4f}")
                logger.info(f"  F1 Score: {f1:.4f}")
                logger.info(f"  ROC AUC: {roc_auc:.4f}")

        return results

    def cross_validate_models(self, X, y, cv=5):
        """Perform cross-validation on all models"""
        cv_results = {}

        for name, model in self.models.items():
            logger.info(f"Cross-validating {name} model...")

            # Define the stratified k-fold with shuffling
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

            # Calculate metrics
            accuracy = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
            precision = cross_val_score(
                model, X, y, cv=skf, scoring='precision')
            recall = cross_val_score(model, X, y, cv=skf, scoring='recall')
            f1 = cross_val_score(model, X, y, cv=skf, scoring='f1')
            roc_auc = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')

            cv_results[name] = {
                'accuracy': (accuracy.mean(), accuracy.std()),
                'precision': (precision.mean(), precision.std()),
                'recall': (recall.mean(), recall.std()),
                'f1': (f1.mean(), f1.std()),
                'roc_auc': (roc_auc.mean(), roc_auc.std())
            }

            logger.info(f"{name} cross-validation results:")
            logger.info(
                f"  Accuracy: {accuracy.mean():.4f} ± {accuracy.std():.4f}")
            logger.info(
                f"  Precision: {precision.mean():.4f} ± {precision.std():.4f}")
            logger.info(f"  Recall: {recall.mean():.4f} ± {recall.std():.4f}")
            logger.info(f"  F1 Score: {f1.mean():.4f} ± {f1.std():.4f}")
            logger.info(
                f"  ROC AUC: {roc_auc.mean():.4f} ± {roc_auc.std():.4f}")

        return cv_results

    def plot_roc_curves(self, results, save_path='results/training/ml_roc_curves.png'):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))

        for name, metrics in results.items():
            plt.plot(metrics['fpr'], metrics['tpr'],
                     label=f"{name.replace('_', ' ').title()} (AUC = {metrics['auc']:.3f})")

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)

        # Save the figure
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        logger.info(f"ROC curves saved to {save_path}")

    def plot_confusion_matrices(self, X_val, y_val, save_path='results/training/ml_confusion_matrices.png'):
        """Plot confusion matrices for all models"""
        # Set up the figure
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models, figsize=(n_models * 5, 5))

        if n_models == 1:
            axes = [axes]

        for ax, (name, model) in zip(axes, self.models.items()):
            # Get predictions
            y_pred = model.predict(X_val)

            # Calculate confusion matrix
            cm = confusion_matrix(y_val, y_pred)

            # Plot
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f"{name.replace('_', ' ').title()}")
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Confusion matrices saved to {save_path}")

    def plot_model_comparison(self, results, save_path='results/training/ml_comparison.png'):
        """Plot comparison of model performances"""
        # Extract metrics
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']

        # Create DataFrame for plotting
        df_metrics = pd.DataFrame(index=models, columns=metrics)
        for name, result in results.items():
            for metric in metrics:
                df_metrics.loc[name, metric] = result[metric]

        # Plot
        plt.figure(figsize=(12, 8))
        df_metrics.plot(kind='bar', figsize=(12, 8))
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.ylim([0, 1])
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower right')
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Model comparison plot saved to {save_path}")

    def plot_all_results(self, results, X_val, y_val, save_dir='results/training'):
        """Generate all plots for the models"""
        os.makedirs(save_dir, exist_ok=True)

        # Plot ROC curves
        self.plot_roc_curves(results, f"{save_dir}/ml_roc_curves.png")

        # Plot confusion matrices
        self.plot_confusion_matrices(
            X_val, y_val, f"{save_dir}/ml_confusion_matrices.png")

        # Plot combined confusion matrix
        self.plot_combined_confusion_matrix(
            X_val, y_val, f"{save_dir}/combined_confusion_matrices.png")

        # Plot model comparison
        self.plot_model_comparison(
            results, f"{save_dir}/ml_comparison.png")

        logger.info(f"All plots saved to {save_dir}")

    def plot_combined_confusion_matrix(self, X_val, y_val, save_path='results/training/combined_confusion_matrices.png'):
        """
        Plot confusion matrices for all models in a single figure grid

        Args:
            X_val: Validation features
            y_val: Validation labels
            save_path: Path to save the combined plot
        """
        # Set up the figure
        n_models = len(self.models)
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

        # Create confusion matrices for each model
        for i, (name, model) in enumerate(self.models.items()):
            if i < len(axes):
                ax = axes[i]

                # Get predictions
                y_pred = model.predict(X_val)

                # Calculate confusion matrix
                cm = confusion_matrix(y_val, y_pred)
                max_value = max(max_value, np.max(cm))

                # Normalize the confusion matrix
                cm_normalized = cm.astype(
                    'float') / cm.sum(axis=1)[:, np.newaxis]

                # Plot
                sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                            ax=ax, cbar=False)

                # Add accuracy value
                accuracy = np.trace(cm) / np.sum(cm)

                # Make the title more descriptive with model name and accuracy
                display_name = name.upper() if name in [
                    'svm', 'knn'] else name.replace('_', ' ').title()
                ax.set_title(
                    f"{display_name}\nAccuracy: {accuracy:.3f}", fontweight='bold')
                ax.set_xlabel('Predicted', fontweight='bold')
                ax.set_ylabel('Actual', fontweight='bold')

        # Hide any unused subplots
        for i in range(n_models, len(axes)):
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
        logger.info(f"Combined confusion matrices saved to {save_path}")

"""
ML Classifier Module
====================
A comprehensive machine learning classification module for multi-omics data.
Provides training, evaluation, and visualization of multiple classifiers.

Usage:
    from ml_classifier import MLClassifier
    
    classifier = MLClassifier()
    results = classifier.train_and_evaluate(X_train, X_test, y_train, y_test)
    classifier.print_results()
    classifier.plot_results()
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any

# Sklearn imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class MLClassifier:
    """
    A comprehensive ML classification module.
    
    Attributes:
        classifiers (dict): Dictionary of trained classifiers
        results (list): List of evaluation results for each classifier
        predictions (dict): Dictionary of predictions for each classifier
        probabilities (dict): Dictionary of prediction probabilities
    """
    
    def __init__(self, random_state: int = 42, include_xgboost: bool = True):
        """
        Initialize the MLClassifier.
        
        Parameters:
            random_state (int): Random state for reproducibility
            include_xgboost (bool): Whether to include XGBoost classifier
        """
        self.random_state = random_state
        self.classifiers = self._create_classifiers(include_xgboost)
        self.results = []
        self.predictions = {}
        self.probabilities = {}
        self.trained = False
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def _create_classifiers(self, include_xgboost: bool) -> Dict[str, Any]:
        """Create default classifiers."""
        classifiers = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, max_depth=10
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, random_state=self.random_state, max_depth=5
            ),
            'SVM (RBF)': SVC(
                kernel='rbf', random_state=self.random_state, probability=True
            ),
            'SVM (Linear)': SVC(
                kernel='linear', random_state=self.random_state, probability=True
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000, random_state=self.random_state, multi_class='ovr'
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=10, random_state=self.random_state
            )
        }
        
        if include_xgboost and XGBOOST_AVAILABLE:
            classifiers['XGBoost'] = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
        
        return classifiers
    
    def add_classifier(self, name: str, classifier: Any) -> None:
        """
        Add a custom classifier.
        
        Parameters:
            name (str): Name of the classifier
            classifier: Sklearn-compatible classifier instance
        """
        self.classifiers[name] = classifier
        
    def remove_classifier(self, name: str) -> None:
        """Remove a classifier by name."""
        if name in self.classifiers:
            del self.classifiers[name]
            
    def _calculate_c_index(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """
        Calculate C-index (Concordance Index) for multi-class classification.
        Uses macro-averaged AUC for multi-class problems.
        """
        try:
            c_index = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
            return c_index
        except Exception:
            return 0.0
    
    def train_and_evaluate(
        self, 
        X_train: np.ndarray, 
        X_test: np.ndarray, 
        y_train: np.ndarray, 
        y_test: np.ndarray,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Train all classifiers and evaluate them.
        
        Parameters:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            verbose: Whether to print progress
            
        Returns:
            List of dictionaries containing evaluation results
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.results = []
        self.predictions = {}
        self.probabilities = {}
        
        if verbose:
            print("Training and evaluating classifiers...")
            print("=" * 80)
        
        for name, clf in self.classifiers.items():
            if verbose:
                print(f"  Training {name}...")
            
            # Train the model
            clf.fit(X_train, y_train)
            
            # Make predictions
            y_pred = clf.predict(X_test)
            y_pred_proba = clf.predict_proba(X_test)
            
            # Store predictions
            self.predictions[name] = y_pred
            self.probabilities[name] = y_pred_proba
            
            # Calculate metrics
            result = {
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision (Macro)': precision_score(y_test, y_pred, average='macro', zero_division=0),
                'Precision (Micro)': precision_score(y_test, y_pred, average='micro', zero_division=0),
                'Recall (Macro)': recall_score(y_test, y_pred, average='macro', zero_division=0),
                'Recall (Micro)': recall_score(y_test, y_pred, average='micro', zero_division=0),
                'F1 (Macro)': f1_score(y_test, y_pred, average='macro', zero_division=0),
                'F1 (Micro)': f1_score(y_test, y_pred, average='micro', zero_division=0),
                'C-Index': self._calculate_c_index(y_test, y_pred_proba)
            }
            
            self.results.append(result)
        
        self.trained = True
        
        if verbose:
            print("=" * 80)
            print(f"✅ Trained and evaluated {len(self.classifiers)} classifiers")
        
        return self.results
    
    def get_results_dataframe(self, sort_by: str = 'Accuracy', ascending: bool = False) -> pd.DataFrame:
        """
        Get results as a pandas DataFrame.
        
        Parameters:
            sort_by: Column to sort by
            ascending: Sort order
            
        Returns:
            DataFrame with all results
        """
        if not self.results:
            raise ValueError("No results available. Run train_and_evaluate first.")
        
        df = pd.DataFrame(self.results)
        return df.sort_values(sort_by, ascending=ascending)
    
    def get_best_model(self, metric: str = 'Accuracy') -> Tuple[str, Any, Dict]:
        """
        Get the best performing model.
        
        Parameters:
            metric: Metric to use for ranking
            
        Returns:
            Tuple of (model_name, model_object, results_dict)
        """
        if not self.results:
            raise ValueError("No results available. Run train_and_evaluate first.")
        
        results_df = self.get_results_dataframe(sort_by=metric, ascending=False)
        best_name = results_df.iloc[0]['Model']
        best_result = results_df.iloc[0].to_dict()
        
        return best_name, self.classifiers[best_name], best_result
    
    def print_results(self, sort_by: str = 'Accuracy') -> None:
        """Print formatted results table."""
        if not self.results:
            print("No results available. Run train_and_evaluate first.")
            return
        
        print("\n" + "=" * 100)
        print("ML CLASSIFIER RESULTS")
        print("=" * 100)
        
        df = self.get_results_dataframe(sort_by=sort_by)
        print(df.to_string(index=False))
        
        # Print best model
        best_name, _, best_result = self.get_best_model(metric=sort_by)
        print("\n" + "-" * 100)
        print(f"🏆 Best Model: {best_name}")
        print(f"   Accuracy: {best_result['Accuracy']:.4f}")
        print(f"   F1 (Macro): {best_result['F1 (Macro)']:.4f}")
        print(f"   C-Index: {best_result['C-Index']:.4f}")
        print("=" * 100)
    
    def get_classification_report(self, model_name: str, target_names: Optional[List[str]] = None) -> str:
        """
        Get classification report for a specific model.
        
        Parameters:
            model_name: Name of the classifier
            target_names: Optional list of class names
            
        Returns:
            Classification report string
        """
        if model_name not in self.predictions:
            raise ValueError(f"Model '{model_name}' not found or not trained.")
        
        return classification_report(
            self.y_test, 
            self.predictions[model_name],
            target_names=target_names,
            digits=4
        )
    
    def get_confusion_matrix(self, model_name: str) -> np.ndarray:
        """Get confusion matrix for a specific model."""
        if model_name not in self.predictions:
            raise ValueError(f"Model '{model_name}' not found or not trained.")
        
        return confusion_matrix(self.y_test, self.predictions[model_name])
    
    def cross_validate(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        cv: int = 5,
        model_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """
        Perform cross-validation on specified models.
        
        Parameters:
            X: Features
            y: Labels
            cv: Number of folds
            model_names: List of model names to evaluate (None = all)
            
        Returns:
            Dictionary of cross-validation results
        """
        cv_results = {}
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        models_to_eval = model_names if model_names else list(self.classifiers.keys())
        
        print(f"\nPerforming {cv}-Fold Cross-Validation...")
        print("-" * 80)
        
        for name in models_to_eval:
            if name not in self.classifiers:
                print(f"Warning: {name} not found in classifiers")
                continue
                
            clf = self.classifiers[name]
            
            acc_scores = cross_val_score(clf, X, y, cv=cv_splitter, scoring='accuracy')
            f1_scores = cross_val_score(clf, X, y, cv=cv_splitter, scoring='f1_macro')
            
            try:
                roc_scores = cross_val_score(clf, X, y, cv=cv_splitter, scoring='roc_auc_ovr')
            except Exception:
                roc_scores = np.array([0.0])
            
            cv_results[name] = {
                'Accuracy': (acc_scores.mean(), acc_scores.std()),
                'F1 (Macro)': (f1_scores.mean(), f1_scores.std()),
                'C-Index': (roc_scores.mean(), roc_scores.std())
            }
            
            print(f"{name}:")
            print(f"  Accuracy: {acc_scores.mean():.4f} (+/- {acc_scores.std():.4f})")
            print(f"  F1-Macro: {f1_scores.mean():.4f} (+/- {f1_scores.std():.4f})")
            print(f"  C-Index:  {roc_scores.mean():.4f} (+/- {roc_scores.std():.4f})")
        
        return cv_results
    
    def plot_results(self, figsize: Tuple[int, int] = (18, 12)) -> None:
        """
        Create comprehensive visualization of results.
        
        Parameters:
            figsize: Figure size
        """
        if not self.results:
            print("No results available. Run train_and_evaluate first.")
            return
        
        results_df = self.get_results_dataframe()
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('ML Classifier Performance Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Accuracy comparison
        bars1 = axes[0, 0].barh(results_df['Model'], results_df['Accuracy'], color='steelblue')
        axes[0, 0].set_xlabel('Accuracy')
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_xlim([0, 1])
        for bar, acc in zip(bars1, results_df['Accuracy']):
            axes[0, 0].text(acc + 0.01, bar.get_y() + bar.get_height()/2, 
                           f'{acc:.3f}', va='center', fontsize=8)
        
        # Plot 2: F1-Score comparison
        x = np.arange(len(results_df))
        width = 0.35
        axes[0, 1].barh(x - width/2, results_df['F1 (Macro)'], width, 
                       label='F1 Macro', color='coral')
        axes[0, 1].barh(x + width/2, results_df['F1 (Micro)'], width, 
                       label='F1 Micro', color='lightgreen')
        axes[0, 1].set_yticks(x)
        axes[0, 1].set_yticklabels(results_df['Model'])
        axes[0, 1].set_xlabel('F1-Score')
        axes[0, 1].set_title('F1-Score Comparison')
        axes[0, 1].legend()
        axes[0, 1].set_xlim([0, 1])
        
        # Plot 3: C-Index comparison
        bars3 = axes[0, 2].barh(results_df['Model'], results_df['C-Index'], color='darkgreen')
        axes[0, 2].set_xlabel('C-Index')
        axes[0, 2].set_title('C-Index Comparison')
        axes[0, 2].set_xlim([0, 1])
        for bar, c_idx in zip(bars3, results_df['C-Index']):
            axes[0, 2].text(c_idx + 0.01, bar.get_y() + bar.get_height()/2, 
                           f'{c_idx:.3f}', va='center', fontsize=8)
        
        # Plot 4: Precision comparison
        axes[1, 0].barh(x - width/2, results_df['Precision (Macro)'], width, 
                       label='Precision Macro', color='purple')
        axes[1, 0].barh(x + width/2, results_df['Precision (Micro)'], width, 
                       label='Precision Micro', color='pink')
        axes[1, 0].set_yticks(x)
        axes[1, 0].set_yticklabels(results_df['Model'])
        axes[1, 0].set_xlabel('Precision')
        axes[1, 0].set_title('Precision Comparison')
        axes[1, 0].legend()
        axes[1, 0].set_xlim([0, 1])
        
        # Plot 5: Recall comparison
        axes[1, 1].barh(x - width/2, results_df['Recall (Macro)'], width, 
                       label='Recall Macro', color='orange')
        axes[1, 1].barh(x + width/2, results_df['Recall (Micro)'], width, 
                       label='Recall Micro', color='cyan')
        axes[1, 1].set_yticks(x)
        axes[1, 1].set_yticklabels(results_df['Model'])
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_title('Recall Comparison')
        axes[1, 1].legend()
        axes[1, 1].set_xlim([0, 1])
        
        # Plot 6: Top models radar chart
        top_3 = results_df.head(3)
        metrics_radar = ['Accuracy', 'F1 (Macro)', 'Precision (Macro)', 'Recall (Macro)', 'C-Index']
        x_pos = np.arange(len(metrics_radar))
        
        for _, row in top_3.iterrows():
            values = [row[metric] for metric in metrics_radar]
            axes[1, 2].plot(x_pos, values, 'o-', label=row['Model'], 
                           linewidth=2, markersize=6, alpha=0.8)
        
        axes[1, 2].set_xticks(x_pos)
        axes[1, 2].set_xticklabels(metrics_radar, rotation=45)
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].set_title('Top 3 Models - All Metrics')
        axes[1, 2].legend()
        axes[1, 2].set_ylim([0, 1])
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrices(self, figsize: Optional[Tuple[int, int]] = None) -> None:
        """Plot confusion matrices for all trained classifiers."""
        if not self.trained:
            print("No models trained. Run train_and_evaluate first.")
            return
        
        n_models = len(self.classifiers)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        if figsize is None:
            figsize = (6 * n_cols, 5 * n_rows)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, (name, clf) in enumerate(self.classifiers.items()):
            cm = self.get_confusion_matrix(name)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{name}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('True')
        
        # Hide empty subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self, figsize: Tuple[int, int] = (10, 8)) -> None:
        """Plot ROC curves for all trained classifiers (micro-averaged for multiclass)."""
        if not self.trained:
            print("No models trained. Run train_and_evaluate first.")
            return
        
        plt.figure(figsize=figsize)
        
        classes = np.unique(self.y_test)
        y_test_bin = label_binarize(self.y_test, classes=classes)
        
        for name, y_pred_proba in self.probabilities.items():
            try:
                # Compute micro-average ROC curve
                fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_pred_proba.ravel())
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})', linewidth=2)
            except Exception as e:
                print(f"Could not plot ROC for {name}: {e}")
        
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves (Micro-averaged for Multiclass)')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def train_classifiers(
    X_train: np.ndarray, 
    X_test: np.ndarray, 
    y_train: np.ndarray, 
    y_test: np.ndarray,
    random_state: int = 42,
    include_xgboost: bool = True,
    verbose: bool = True
) -> MLClassifier:
    """
    Convenience function to train all classifiers and return results.
    
    Parameters:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        random_state: Random state for reproducibility
        include_xgboost: Whether to include XGBoost
        verbose: Whether to print progress
        
    Returns:
        Trained MLClassifier object with all results
    """
    classifier = MLClassifier(random_state=random_state, include_xgboost=include_xgboost)
    classifier.train_and_evaluate(X_train, X_test, y_train, y_test, verbose=verbose)
    return classifier


# For backwards compatibility
def get_default_classifiers(random_state: int = 42) -> Dict[str, Any]:
    """Get dictionary of default classifiers."""
    clf = MLClassifier(random_state=random_state)
    return clf.classifiers

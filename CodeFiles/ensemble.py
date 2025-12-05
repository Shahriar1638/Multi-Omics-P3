"""
Ensemble Models for Multi-Omics Cancer Classification
======================================================
This module provides various ensemble methods for combining predictions
from multiple classifiers for cancer subtype classification.

Usage:
------
    from ensemble import (
        VotingEnsemble, 
        StackingEnsemble, 
        WeightedAverageEnsemble,
        BaggingEnsemble,
        BoostingEnsemble,
        evaluate_ensemble,
        create_default_classifiers,
        create_all_ensembles,
        evaluate_multiple_ensembles,
        print_ensemble_comparison,
        plot_ensemble_comparison
    )

Example:
--------
    # Create base classifiers
    classifiers = create_default_classifiers()
    
    # Voting Ensemble
    voting = VotingEnsemble(classifiers, voting='soft')
    voting.fit(X_train, y_train)
    predictions = voting.predict(X_test)
    
    # Evaluate
    results = evaluate_ensemble(voting, X_test, y_test)
    print(results)
    
    # Or create all ensembles at once
    ensembles = create_all_ensembles(X_train, y_train)
    results_df = evaluate_multiple_ensembles(ensembles, X_test, y_test)
    print_ensemble_comparison(results_df)
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
    BaggingClassifier,
    AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, ClassifierMixin
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
import warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def create_default_classifiers(random_state=42):
    """
    Create a dictionary of default classifiers for ensemble methods.
    
    Returns:
    --------
    dict: Dictionary of classifier name -> classifier instance
    """
    classifiers = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=random_state
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, 
            max_depth=5, 
            random_state=random_state
        ),
        'SVM (RBF)': SVC(
            kernel='rbf', 
            probability=True, 
            random_state=random_state
        ),
        'SVM (Linear)': SVC(
            kernel='linear', 
            probability=True, 
            random_state=random_state
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000, 
            random_state=random_state, 
            multi_class='ovr'
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10, 
            random_state=random_state
        )
    }
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        classifiers['XGBoost'] = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.001,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
    
    return classifiers


def concordance_index(y_true, y_pred_proba):
    """
    Calculate C-index (Concordance Index) for multi-class classification.
    Uses macro-averaged AUC for multi-class problems.
    
    Parameters:
    -----------
    y_true : array-like
        True class labels
    y_pred_proba : array-like
        Predicted probabilities for each class
    
    Returns:
    --------
    float: C-index score
    """
    try:
        return roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
    except:
        return 0.0


def evaluate_ensemble(model, X_test, y_test, model_name="Ensemble"):
    """
    Comprehensive evaluation of an ensemble model.
    
    Parameters:
    -----------
    model : classifier
        Trained classifier with predict and predict_proba methods
    X_test : array-like
        Test features
    y_test : array-like
        True test labels
    model_name : str
        Name of the model for display
    
    Returns:
    --------
    dict: Dictionary containing all evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    try:
        y_pred_proba = model.predict_proba(X_test)
        c_index = concordance_index(y_test, y_pred_proba)
    except:
        c_index = 0.0
    
    results = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision (Macro)': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'Precision (Micro)': precision_score(y_test, y_pred, average='micro', zero_division=0),
        'Recall (Macro)': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'Recall (Micro)': recall_score(y_test, y_pred, average='micro', zero_division=0),
        'F1 (Macro)': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'F1 (Micro)': f1_score(y_test, y_pred, average='micro', zero_division=0),
        'C-Index': c_index
    }
    
    return results


def evaluate_multiple_ensembles(ensembles, X_test, y_test):
    """
    Evaluate multiple ensemble models and return a comparison DataFrame.
    
    Parameters:
    -----------
    ensembles : dict
        Dictionary of ensemble name -> trained ensemble model
    X_test : array-like
        Test features
    y_test : array-like
        True test labels
    
    Returns:
    --------
    pd.DataFrame: DataFrame with evaluation metrics for all ensembles
    """
    results = []
    for name, model in ensembles.items():
        result = evaluate_ensemble(model, X_test, y_test, model_name=name)
        results.append(result)
    
    return pd.DataFrame(results).sort_values('Accuracy', ascending=False)


# ==============================================================================
# ENSEMBLE METHODS
# ==============================================================================

class VotingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Voting Ensemble Classifier.
    
    Combines multiple classifiers using voting strategy.
    
    Parameters:
    -----------
    classifiers : dict
        Dictionary of classifier name -> classifier instance
    voting : str, default='soft'
        'hard' for majority voting, 'soft' for probability averaging
    weights : list, optional
        Weights for each classifier (for weighted voting)
    """
    
    def __init__(self, classifiers, voting='soft', weights=None):
        self.classifiers = classifiers
        self.voting = voting
        self.weights = weights
        self.fitted_classifiers_ = {}
        self.classes_ = None
    
    def fit(self, X, y):
        """Fit all base classifiers."""
        self.classes_ = np.unique(y)
        
        for name, clf in self.classifiers.items():
            print(f"Training {name}...")
            clf.fit(X, y)
            self.fitted_classifiers_[name] = clf
        
        return self
    
    def predict(self, X):
        """Predict using voting strategy."""
        if self.voting == 'hard':
            # Majority voting
            predictions = np.array([clf.predict(X) for clf in self.fitted_classifiers_.values()])
            # Use mode for each sample
            from scipy import stats
            predictions = stats.mode(predictions, axis=0, keepdims=False)[0]
            return predictions
        else:
            # Soft voting (probability averaging)
            probas = self.predict_proba(X)
            return self.classes_[np.argmax(probas, axis=1)]
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        probas = []
        weights = self.weights if self.weights else [1] * len(self.fitted_classifiers_)
        
        for (name, clf), weight in zip(self.fitted_classifiers_.items(), weights):
            try:
                proba = clf.predict_proba(X) * weight
                probas.append(proba)
            except:
                # If predict_proba not available, use one-hot encoding of predictions
                preds = clf.predict(X)
                proba = np.zeros((len(X), len(self.classes_)))
                for i, pred in enumerate(preds):
                    proba[i, pred] = 1.0 * weight
                probas.append(proba)
        
        # Average probabilities
        avg_proba = np.mean(probas, axis=0)
        # Normalize
        avg_proba /= avg_proba.sum(axis=1, keepdims=True)
        return avg_proba


class StackingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Stacking Ensemble Classifier.
    
    Uses base classifier predictions as features for a meta-classifier.
    
    Parameters:
    -----------
    base_classifiers : dict
        Dictionary of classifier name -> classifier instance
    meta_classifier : classifier, optional
        Meta-classifier to combine base predictions. Default: LogisticRegression
    use_probas : bool, default=True
        Whether to use probability outputs from base classifiers
    cv : int, default=5
        Number of cross-validation folds for generating meta-features
    """
    
    def __init__(self, base_classifiers, meta_classifier=None, use_probas=True, cv=5):
        self.base_classifiers = base_classifiers
        self.meta_classifier = meta_classifier or LogisticRegression(max_iter=1000)
        self.use_probas = use_probas
        self.cv = cv
        self.fitted_base_classifiers_ = {}
        self.classes_ = None
    
    def _generate_meta_features(self, X, y=None, is_train=True):
        """Generate meta-features from base classifier predictions."""
        meta_features = []
        
        if is_train:
            # Use cross-validation to generate out-of-fold predictions
            from sklearn.model_selection import cross_val_predict
            
            for name, clf in self.base_classifiers.items():
                if self.use_probas:
                    try:
                        proba = cross_val_predict(clf, X, y, cv=self.cv, method='predict_proba')
                        meta_features.append(proba)
                    except:
                        preds = cross_val_predict(clf, X, y, cv=self.cv)
                        # One-hot encode predictions
                        proba = np.zeros((len(X), len(self.classes_)))
                        for i, pred in enumerate(preds):
                            proba[i, pred] = 1.0
                        meta_features.append(proba)
                else:
                    preds = cross_val_predict(clf, X, y, cv=self.cv)
                    meta_features.append(preds.reshape(-1, 1))
        else:
            # Use fitted classifiers for test predictions
            for name, clf in self.fitted_base_classifiers_.items():
                if self.use_probas:
                    try:
                        proba = clf.predict_proba(X)
                        meta_features.append(proba)
                    except:
                        preds = clf.predict(X)
                        proba = np.zeros((len(X), len(self.classes_)))
                        for i, pred in enumerate(preds):
                            proba[i, pred] = 1.0
                        meta_features.append(proba)
                else:
                    preds = clf.predict(X)
                    meta_features.append(preds.reshape(-1, 1))
        
        return np.hstack(meta_features)
    
    def fit(self, X, y):
        """Fit base classifiers and meta-classifier."""
        self.classes_ = np.unique(y)
        
        # Generate meta-features using cross-validation
        print("Generating meta-features using cross-validation...")
        meta_X = self._generate_meta_features(X, y, is_train=True)
        
        # Fit base classifiers on full training data
        print("Fitting base classifiers on full training data...")
        for name, clf in self.base_classifiers.items():
            print(f"  Training {name}...")
            clf.fit(X, y)
            self.fitted_base_classifiers_[name] = clf
        
        # Fit meta-classifier
        print("Fitting meta-classifier...")
        self.meta_classifier.fit(meta_X, y)
        
        return self
    
    def predict(self, X):
        """Predict using stacked ensemble."""
        meta_X = self._generate_meta_features(X, is_train=False)
        return self.meta_classifier.predict(meta_X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        meta_X = self._generate_meta_features(X, is_train=False)
        try:
            return self.meta_classifier.predict_proba(meta_X)
        except:
            preds = self.meta_classifier.predict(meta_X)
            proba = np.zeros((len(X), len(self.classes_)))
            for i, pred in enumerate(preds):
                proba[i, pred] = 1.0
            return proba


class WeightedAverageEnsemble(BaseEstimator, ClassifierMixin):
    """
    Weighted Average Ensemble Classifier.
    
    Combines classifiers using learned or specified weights based on performance.
    
    Parameters:
    -----------
    classifiers : dict
        Dictionary of classifier name -> classifier instance
    weights : dict or str, default='accuracy'
        - dict: Custom weights for each classifier
        - 'accuracy': Learn weights based on validation accuracy
        - 'f1': Learn weights based on validation F1 score
        - 'equal': Equal weights for all classifiers
    validation_split : float, default=0.2
        Proportion of training data to use for weight learning
    """
    
    def __init__(self, classifiers, weights='accuracy', validation_split=0.2):
        self.classifiers = classifiers
        self.weights = weights
        self.validation_split = validation_split
        self.fitted_classifiers_ = {}
        self.learned_weights_ = {}
        self.classes_ = None
    
    def fit(self, X, y):
        """Fit classifiers and learn weights."""
        from sklearn.model_selection import train_test_split
        
        self.classes_ = np.unique(y)
        
        # Split for weight learning
        if isinstance(self.weights, str) and self.weights in ['accuracy', 'f1']:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.validation_split, random_state=42, stratify=y
            )
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
        
        # Fit classifiers
        scores = {}
        for name, clf in self.classifiers.items():
            print(f"Training {name}...")
            clf.fit(X_train, y_train)
            self.fitted_classifiers_[name] = clf
            
            # Calculate validation score for weight learning
            if X_val is not None:
                preds = clf.predict(X_val)
                if self.weights == 'accuracy':
                    scores[name] = accuracy_score(y_val, preds)
                elif self.weights == 'f1':
                    scores[name] = f1_score(y_val, preds, average='macro')
        
        # Set weights
        if isinstance(self.weights, dict):
            self.learned_weights_ = self.weights
        elif self.weights == 'equal':
            self.learned_weights_ = {name: 1.0 for name in self.classifiers}
        else:
            # Normalize scores to get weights
            total_score = sum(scores.values())
            self.learned_weights_ = {name: score / total_score for name, score in scores.items()}
        
        print(f"\nLearned weights: {self.learned_weights_}")
        
        # Refit on full data if we used validation split
        if X_val is not None:
            for name, clf in self.classifiers.items():
                clf.fit(X, y)
                self.fitted_classifiers_[name] = clf
        
        return self
    
    def predict(self, X):
        """Predict using weighted average probabilities."""
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]
    
    def predict_proba(self, X):
        """Predict weighted average class probabilities."""
        weighted_probas = np.zeros((len(X), len(self.classes_)))
        total_weight = sum(self.learned_weights_.values())
        
        for name, clf in self.fitted_classifiers_.items():
            weight = self.learned_weights_.get(name, 1.0)
            try:
                proba = clf.predict_proba(X)
            except:
                preds = clf.predict(X)
                proba = np.zeros((len(X), len(self.classes_)))
                for i, pred in enumerate(preds):
                    proba[i, pred] = 1.0
            
            weighted_probas += proba * weight
        
        # Normalize
        weighted_probas /= total_weight
        return weighted_probas


class BaggingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Custom Bagging Ensemble Classifier.
    
    Trains multiple instances of the same base classifier on bootstrap samples.
    
    Parameters:
    -----------
    base_classifier : classifier
        Base classifier to bag
    n_estimators : int, default=10
        Number of base classifiers
    max_samples : float, default=0.8
        Proportion of samples for each bootstrap
    random_state : int, default=42
        Random seed for reproducibility
    """
    
    def __init__(self, base_classifier=None, n_estimators=10, max_samples=0.8, random_state=42):
        if base_classifier is None:
            self.base_classifier = RandomForestClassifier(n_estimators=50, random_state=random_state)
        else:
            self.base_classifier = base_classifier
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.estimators_ = []
        self.classes_ = None
    
    def fit(self, X, y):
        """Fit bagged classifiers."""
        from sklearn.base import clone
        
        self.classes_ = np.unique(y)
        np.random.seed(self.random_state)
        
        n_samples = int(len(X) * self.max_samples)
        
        for i in range(self.n_estimators):
            print(f"Training estimator {i+1}/{self.n_estimators}...")
            # Bootstrap sample
            indices = np.random.choice(len(X), size=n_samples, replace=True)
            X_boot = X[indices] if isinstance(X, np.ndarray) else X.iloc[indices]
            y_boot = y[indices] if isinstance(y, np.ndarray) else y.iloc[indices]
            
            # Clone and fit
            estimator = clone(self.base_classifier)
            estimator.fit(X_boot, y_boot)
            self.estimators_.append(estimator)
        
        return self
    
    def predict(self, X):
        """Predict using majority voting from bagged classifiers."""
        predictions = np.array([est.predict(X) for est in self.estimators_])
        from scipy import stats
        return stats.mode(predictions, axis=0, keepdims=False)[0]
    
    def predict_proba(self, X):
        """Predict averaged class probabilities."""
        probas = []
        for est in self.estimators_:
            try:
                probas.append(est.predict_proba(X))
            except:
                preds = est.predict(X)
                proba = np.zeros((len(X), len(self.classes_)))
                for i, pred in enumerate(preds):
                    proba[i, pred] = 1.0
                probas.append(proba)
        
        return np.mean(probas, axis=0)


class BoostingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Custom Boosting Ensemble using AdaBoost methodology.
    
    Parameters:
    -----------
    base_classifier : classifier
        Base weak classifier
    n_estimators : int, default=50
        Number of boosting iterations
    learning_rate : float, default=1.0
        Weight applied to each classifier at each iteration
    random_state : int, default=42
        Random seed
    """
    
    def __init__(self, base_classifier=None, n_estimators=50, learning_rate=1.0, random_state=42):
        self.base_classifier = base_classifier or DecisionTreeClassifier(max_depth=3, random_state=random_state)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.ada_boost_ = None
        self.classes_ = None
    
    def fit(self, X, y):
        """Fit boosting ensemble."""
        from sklearn.base import clone
        
        self.classes_ = np.unique(y)
        
        # Use sklearn's AdaBoost with our base classifier
        self.ada_boost_ = AdaBoostClassifier(
            estimator=clone(self.base_classifier),
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            algorithm='SAMME'
        )
        
        print("Training AdaBoost ensemble...")
        self.ada_boost_.fit(X, y)
        
        return self
    
    def predict(self, X):
        """Predict class labels."""
        return self.ada_boost_.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.ada_boost_.predict_proba(X)


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def create_all_ensembles(X_train, y_train, classifiers=None, random_state=42):
    """
    Create and train all ensemble types for comparison.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    classifiers : dict, optional
        Custom classifiers. If None, default classifiers are used.
    random_state : int
        Random seed
    
    Returns:
    --------
    dict: Dictionary of ensemble name -> trained ensemble
    """
    if classifiers is None:
        classifiers = create_default_classifiers(random_state)
    
    ensembles = {}
    
    # Voting ensembles
    print("\n" + "="*60)
    print("Training Voting Ensemble (Soft)")
    print("="*60)
    voting_soft = VotingEnsemble(classifiers.copy(), voting='soft')
    voting_soft.fit(X_train, y_train)
    ensembles['Voting (Soft)'] = voting_soft
    
    print("\n" + "="*60)
    print("Training Voting Ensemble (Hard)")
    print("="*60)
    voting_hard = VotingEnsemble(create_default_classifiers(random_state), voting='hard')
    voting_hard.fit(X_train, y_train)
    ensembles['Voting (Hard)'] = voting_hard
    
    # Stacking ensemble
    print("\n" + "="*60)
    print("Training Stacking Ensemble")
    print("="*60)
    stacking = StackingEnsemble(
        create_default_classifiers(random_state),
        meta_classifier=LogisticRegression(max_iter=1000)
    )
    stacking.fit(X_train, y_train)
    ensembles['Stacking'] = stacking
    
    # Weighted ensemble
    print("\n" + "="*60)
    print("Training Weighted Average Ensemble")
    print("="*60)
    weighted = WeightedAverageEnsemble(
        create_default_classifiers(random_state),
        weights='accuracy'
    )
    weighted.fit(X_train, y_train)
    ensembles['Weighted Average'] = weighted
    
    # Bagging ensemble
    print("\n" + "="*60)
    print("Training Bagging Ensemble")
    print("="*60)
    bagging = BaggingEnsemble(
        RandomForestClassifier(n_estimators=50, random_state=random_state),
        n_estimators=10
    )
    bagging.fit(X_train, y_train)
    ensembles['Bagging'] = bagging
    
    # Boosting ensemble
    print("\n" + "="*60)
    print("Training Boosting Ensemble")
    print("="*60)
    boosting = BoostingEnsemble(n_estimators=50, random_state=random_state)
    boosting.fit(X_train, y_train)
    ensembles['Boosting (AdaBoost)'] = boosting
    
    return ensembles


def print_ensemble_comparison(results_df):
    """
    Print a formatted comparison of ensemble results.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame from evaluate_multiple_ensembles
    """
    print("\n" + "="*100)
    print("ENSEMBLE MODEL COMPARISON")
    print("="*100)
    
    print("\n📊 Sorted by Accuracy:")
    print("-"*100)
    print(results_df.sort_values('Accuracy', ascending=False).to_string(index=False))
    
    print("\n📊 Sorted by C-Index:")
    print("-"*100)
    print(results_df.sort_values('C-Index', ascending=False)[['Model', 'Accuracy', 'F1 (Macro)', 'C-Index']].to_string(index=False))
    
    # Best model summary
    best_acc = results_df.loc[results_df['Accuracy'].idxmax()]
    best_f1 = results_df.loc[results_df['F1 (Macro)'].idxmax()]
    best_cindex = results_df.loc[results_df['C-Index'].idxmax()]
    
    print("\n🏆 BEST MODELS:")
    print("-"*50)
    print(f"  Best Accuracy:  {best_acc['Model']} ({best_acc['Accuracy']:.4f})")
    print(f"  Best F1 (Macro): {best_f1['Model']} ({best_f1['F1 (Macro)']:.4f})")
    print(f"  Best C-Index:   {best_cindex['Model']} ({best_cindex['C-Index']:.4f})")


# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def plot_ensemble_comparison(results_df, save_path=None):
    """
    Create visualization comparing ensemble methods.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame from evaluate_multiple_ensembles
    save_path : str, optional
        Path to save the figure
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Sort by accuracy for consistent ordering
    results_sorted = results_df.sort_values('Accuracy', ascending=True)
    
    # Plot 1: Accuracy comparison
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results_sorted)))
    axes[0, 0].barh(results_sorted['Model'], results_sorted['Accuracy'], color=colors)
    axes[0, 0].set_xlabel('Accuracy')
    axes[0, 0].set_title('Ensemble Accuracy Comparison')
    axes[0, 0].set_xlim([0, 1])
    for i, (acc, model) in enumerate(zip(results_sorted['Accuracy'], results_sorted['Model'])):
        axes[0, 0].text(acc + 0.01, i, f'{acc:.3f}', va='center', fontsize=9)
    
    # Plot 2: F1 Score comparison
    axes[0, 1].barh(results_sorted['Model'], results_sorted['F1 (Macro)'], color=colors)
    axes[0, 1].set_xlabel('F1 Score (Macro)')
    axes[0, 1].set_title('Ensemble F1 Score Comparison')
    axes[0, 1].set_xlim([0, 1])
    for i, (f1, model) in enumerate(zip(results_sorted['F1 (Macro)'], results_sorted['Model'])):
        axes[0, 1].text(f1 + 0.01, i, f'{f1:.3f}', va='center', fontsize=9)
    
    # Plot 3: C-Index comparison
    axes[1, 0].barh(results_sorted['Model'], results_sorted['C-Index'], color=colors)
    axes[1, 0].set_xlabel('C-Index')
    axes[1, 0].set_title('Ensemble C-Index Comparison')
    axes[1, 0].set_xlim([0, 1])
    for i, (ci, model) in enumerate(zip(results_sorted['C-Index'], results_sorted['Model'])):
        axes[1, 0].text(ci + 0.01, i, f'{ci:.3f}', va='center', fontsize=9)
    
    # Plot 4: Multi-metric comparison for top models
    top_5 = results_df.nlargest(5, 'Accuracy')
    metrics = ['Accuracy', 'F1 (Macro)', 'Precision (Macro)', 'Recall (Macro)', 'C-Index']
    x_pos = np.arange(len(metrics))
    
    for i, (_, row) in enumerate(top_5.iterrows()):
        values = [row[metric] for metric in metrics]
        axes[1, 1].plot(x_pos, values, 'o-', label=row['Model'], linewidth=2, markersize=6)
    
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(metrics, rotation=45)
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Top 5 Ensembles - Multi-Metric Comparison')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
    # Generate sample data
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=500, n_features=50, n_informative=20,
                               n_classes=3, n_clusters_per_class=1, random_state=42)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                         random_state=42, stratify=y)
    
    # Create and evaluate all ensembles
    ensembles = create_all_ensembles(X_train, y_train)
    
    # Evaluate
    results = evaluate_multiple_ensembles(ensembles, X_test, y_test)
    
    # Print comparison
    print_ensemble_comparison(results)
    
    # Plot comparison
    plot_ensemble_comparison(results)
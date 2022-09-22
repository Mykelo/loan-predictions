from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class LogisticRegressionRF(BaseEstimator, ClassifierMixin):
    """
    Custom model responsible for calculating feature importances using
    Random Forest and using them to select top N features before training
    Logistic regression.
    """

    def __init__(self, random_state: Optional[int] = 0, features_num: int = 100, max_depth: int = 3):
        self.random_state = random_state
        self.features_num = features_num
        self.max_depth = max_depth
        self.logistic_regression = LogisticRegression(random_state=random_state)
        self.random_forest = RandomForestClassifier(random_state=random_state, max_depth=max_depth)
        self.top_features: Optional[np.ndarray] = None

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.random_forest.fit(X, y)
        # Find the most important features
        self.top_features = np.argsort(self.random_forest.feature_importances_)[-self.features_num:]

        self.logistic_regression.fit(X[:, self.top_features], y)

        # Return the classifier
        return self

    def predict(self, X):

        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        X = X[:, self.top_features]
        
        return self.logistic_regression.predict(X)
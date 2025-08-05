"""Ensemble Anomaly Detection Module"""

# Copyright 2025 Netskope, Inc.
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Authors:
# - Colin Estep
# - Dagmawi Mulugeta

import logging
import pickle
from typing import Dict, Optional

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# TensorFlow support for autoencoder
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

from .utils import safe_create_path


# AutoencoderAnomalyDetector removed - TensorFlow disabled
class AutoencoderAnomalyDetector:
    """
    Autoencoder-based anomaly detection for network traffic patterns.
    """

    def __init__(self, encoding_dim: int = 32, contamination: float = 0.1):
        """
        Initialize the autoencoder anomaly detector.

        Args:
            encoding_dim (int): Dimension of the encoded representation
            contamination (float): Expected proportion of outliers (for threshold setting)
        """
        if not HAS_TENSORFLOW:
            raise ImportError(
                "TensorFlow is required for AutoencoderAnomalyDetector but is not installed"
            )

        # Set random seeds for full reproducibility across the entire ensemble
        import random
        random.seed(42)  # Python's built-in random module
        np.random.seed(42)  # NumPy random operations (used by scikit-learn)
        tf.random.set_seed(42)  # TensorFlow operations
        
        self.encoding_dim = encoding_dim
        self.contamination = contamination
        self.autoencoder = None
        self.encoder = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.input_dim = None
        self.is_fitted = False

    def _build_autoencoder(self, input_dim: int) -> None:
        """Build the autoencoder architecture."""
        # Input layer
        input_layer = layers.Input(shape=(input_dim,))

        # Encoder
        encoded = layers.Dense(64, activation="relu")(input_layer)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(self.encoding_dim, activation="relu")(encoded)

        # Decoder
        decoded = layers.Dense(64, activation="relu")(encoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(input_dim, activation="linear")(decoded)

        # Models
        self.autoencoder = models.Model(input_layer, decoded)
        self.encoder = models.Model(input_layer, encoded)

        # Compile
        self.autoencoder.compile(
            optimizer=optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"]
        )

    def fit(
        self, X: np.ndarray, epochs: int = 100, validation_split: float = 0.2
    ) -> "AutoencoderAnomalyDetector":
        """
        Train the autoencoder on normal data.

        Args:
            X (np.ndarray): Training data (assumed to be mostly normal)
            epochs (int): Number of training epochs
            validation_split (float): Fraction of data to use for validation

        Returns:
            Self for method chaining
        """
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        self.input_dim = X.shape[1]

        # Build the model
        self._build_autoencoder(self.input_dim)

        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )

        # Train the autoencoder
        history = self.autoencoder.fit(
            X_scaled,
            X_scaled,
            epochs=epochs,
            batch_size=32,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=0,
        )

        # Calculate reconstruction errors on training data to set threshold
        reconstructed = self.autoencoder.predict(X_scaled, verbose=0)
        reconstruction_errors = np.mean(np.square(X_scaled - reconstructed), axis=1)

        # Set threshold based on contamination level
        self.threshold = np.percentile(
            reconstruction_errors, (1 - self.contamination) * 100
        )

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies in the data.

        Args:
            X (np.ndarray): Data to predict on

        Returns:
            np.ndarray: Predictions (1 for normal, -1 for anomaly)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X_scaled = self.scaler.transform(X)
        reconstructed = self.autoencoder.predict(X_scaled, verbose=0)
        reconstruction_errors = np.mean(np.square(X_scaled - reconstructed), axis=1)

        # Return sklearn-style predictions (1 for normal, -1 for anomaly)
        return np.where(reconstruction_errors <= self.threshold, 1, -1)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Return anomaly scores (reconstruction errors).

        Args:
            X (np.ndarray): Data to score

        Returns:
            np.ndarray: Anomaly scores (higher = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing scores")

        X_scaled = self.scaler.transform(X)
        reconstructed = self.autoencoder.predict(X_scaled, verbose=0)
        reconstruction_errors = np.mean(np.square(X_scaled - reconstructed), axis=1)

        # Return negative scores to match sklearn convention (lower = more anomalous)
        return -reconstruction_errors


class EnsembleAnomalyDetector:
    """
    Ensemble of multiple anomaly detection algorithms for robust supply chain compromise detection.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        isolation_forest_params: Optional[Dict] = None,
        one_class_svm_params: Optional[Dict] = None,
        autoencoder_params: Optional[Dict] = None,
        use_adaptive_threshold: bool = False,
    ):
        """
        Initialize the ensemble anomaly detector.

        Args:
            contamination (float): Expected proportion of outliers
            isolation_forest_params (dict): Parameters for Isolation Forest
            one_class_svm_params (dict): Parameters for One-Class SVM
            autoencoder_params (dict): Parameters for Autoencoder
            use_adaptive_threshold (bool): If True, use adaptive threshold based on training data
        """
        self.contamination = contamination
        self.use_adaptive_threshold = use_adaptive_threshold
        self.adaptive_threshold = None
        self.logger = logging.getLogger(__name__)
        
        # Set random seeds for full reproducibility across the entire ensemble
        import random
        random.seed(42)  # Python's built-in random module
        np.random.seed(42)  # NumPy random operations (used by scikit-learn)

        # Default parameters
        if isolation_forest_params is None:
            isolation_forest_params = {
                "n_estimators": 100,
                "contamination": contamination,
                "random_state": 42,
                "n_jobs": -1,
            }

        if one_class_svm_params is None:
            one_class_svm_params = {
                "nu": contamination,
                "gamma": "scale",
                "kernel": "rbf",
            }

        if autoencoder_params is None:
            # Use a lower contamination for autoencoder to reduce false positives
            # 0.01 = 1% expected anomalies (more reasonable than 10%)
            autoencoder_params = {"encoding_dim": 32, "contamination": 0.01}

        # Initialize models
        self.isolation_forest = IsolationForest(**isolation_forest_params)
        self.one_class_svm = Pipeline(
            [("scaler", StandardScaler()), ("svm", OneClassSVM(**one_class_svm_params))]
        )
        
        # Initialize autoencoder only if TensorFlow is available
        if HAS_TENSORFLOW:
            self.autoencoder = AutoencoderAnomalyDetector(**autoencoder_params)
            # Ensemble weights (can be learned or set manually)
            self.weights = np.array([0.4, 0.3, 0.3])  # IF, SVM, Autoencoder
        else:
            self.autoencoder = None
            # Adjust weights when autoencoder is not available
            self.weights = np.array([0.5, 0.5])  # IF, SVM only
            self.logger.warning(
                "TensorFlow not available. Ensemble will use only Isolation Forest and One-Class SVM."
            )
        self.is_fitted = False

    def fit(
        self, X: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> "EnsembleAnomalyDetector":
        """
        Fit all ensemble models on the training data.

        Args:
            X (np.ndarray): Training data (assumed to be mostly normal)
            sample_weight (np.ndarray): Sample weights (not used by all models)

        Returns:
            Self for method chaining
        """
        self.logger.info(
            f"Training ensemble anomaly detector on {X.shape[0]} samples with {X.shape[1]} features"
        )

        # Fit Isolation Forest
        self.logger.info("Training Isolation Forest...")
        self.isolation_forest.fit(X, sample_weight=sample_weight)

        # Fit One-Class SVM
        self.logger.info("Training One-Class SVM...")
        self.one_class_svm.fit(X)

        # Fit Autoencoder (if TensorFlow is available)
        if HAS_TENSORFLOW and self.autoencoder is not None:
            self.logger.info("Training Autoencoder...")
            try:
                self.autoencoder.fit(X)
            except Exception as e:
                self.logger.warning(
                    f"Autoencoder training failed: {e}. Continuing with other models."
                )
                # Disable autoencoder if training fails
                self.autoencoder = None
                self.weights = np.array([0.5, 0.5])  # Revert to IF, SVM only
        else:
            self.logger.info("TensorFlow not available, skipping autoencoder training")

        # Mark as fitted before computing adaptive threshold
        self.is_fitted = True
        
        # If using adaptive threshold, compute it based on training data scores
        if self.use_adaptive_threshold:
            self.logger.info("Computing adaptive threshold based on training data")
            train_scores = self.decision_function(X)
            # Set threshold to be much more lenient than the minimum training score
            # This ensures no training samples are classified as anomalies
            # Use a large margin to account for any numerical variations
            self.adaptive_threshold = np.min(train_scores) - abs(np.min(train_scores)) * 0.1 - 10.0
            self.logger.info(f"Adaptive threshold set to: {self.adaptive_threshold}")
        
        self.logger.info("Ensemble training completed")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies using ensemble voting.

        Args:
            X (np.ndarray): Data to predict on

        Returns:
            np.ndarray: Predictions (1 for normal, -1 for anomaly)
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")

        # Get predictions from each model
        predictions = []

        # Isolation Forest
        if_pred = self.isolation_forest.predict(X)
        predictions.append(if_pred)

        # One-Class SVM
        svm_pred = self.one_class_svm.predict(X)
        predictions.append(svm_pred)

        # Autoencoder (if available and trained)
        if HAS_TENSORFLOW and self.autoencoder is not None and self.autoencoder.is_fitted:
            try:
                ae_pred = self.autoencoder.predict(X)
                predictions.append(ae_pred)
            except Exception as e:
                self.logger.warning(f"Autoencoder prediction failed: {e}")
                # Don't append anything if autoencoder fails

        # If using adaptive threshold, use decision scores instead of votes
        if self.use_adaptive_threshold and self.adaptive_threshold is not None:
            scores = self.decision_function(X)
            return np.where(scores > self.adaptive_threshold, 1, -1)
        
        # Otherwise use weighted ensemble voting
        predictions = np.array(predictions)
        weighted_scores = np.average(predictions, axis=0, weights=self.weights)

        # Convert to binary predictions
        return np.where(weighted_scores > 0, 1, -1)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Return ensemble anomaly scores.

        Args:
            X (np.ndarray): Data to score

        Returns:
            np.ndarray: Anomaly scores (lower = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before computing scores")

        # Get decision scores from each model
        scores = []

        # Isolation Forest
        if_scores = self.isolation_forest.decision_function(X)
        scores.append(if_scores)

        # One-Class SVM
        svm_scores = self.one_class_svm.decision_function(X)
        scores.append(svm_scores)

        # Autoencoder (if available and trained)
        if HAS_TENSORFLOW and self.autoencoder is not None and self.autoencoder.is_fitted:
            try:
                ae_scores = self.autoencoder.decision_function(X)
                scores.append(ae_scores)
            except Exception as e:
                self.logger.warning(f"Autoencoder scoring failed: {e}")
                # Don't append anything if autoencoder fails

        # Weighted ensemble scoring
        scores = np.array(scores)
        weighted_scores = np.average(scores, axis=0, weights=self.weights)

        return weighted_scores

    def predict_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Alias for decision_function to match expected interface.
        
        Args:
            X (np.ndarray): Data to score
            
        Returns:
            np.ndarray: Anomaly scores (lower = more anomalous)
        """
        return self.decision_function(X)

    def save_model(self, model_path: str) -> None:
        """
        Save the trained ensemble model.

        Args:
            model_path (str): Path to save the model
        """
        safe_create_path(model_path)

        # Save the ensemble (excluding TensorFlow components)
        ensemble_data = {
            "isolation_forest": self.isolation_forest,
            "one_class_svm": self.one_class_svm,
            "weights": self.weights,
            "contamination": self.contamination,
            "is_fitted": self.is_fitted,
            "use_adaptive_threshold": self.use_adaptive_threshold,
            "adaptive_threshold": self.adaptive_threshold,
        }

        with open(model_path, "wb") as f:
            pickle.dump(ensemble_data, f)

        # Save autoencoder separately if trained
        if len(self.weights) > 2 and self.weights[2] > 0 and self.autoencoder is not None and self.autoencoder.is_fitted:
            autoencoder_path = model_path.replace(".pkl", "_autoencoder.h5")
            self.autoencoder.autoencoder.save(autoencoder_path)

        self.logger.info(f"Ensemble model saved to {model_path}")

    def load_model(self, model_path: str) -> "EnsembleAnomalyDetector":
        """
        Load a trained ensemble model.

        Args:
            model_path (str): Path to the saved model

        Returns:
            Self for method chaining
        """
        with open(model_path, "rb") as f:
            ensemble_data = pickle.load(f)

        self.isolation_forest = ensemble_data["isolation_forest"]
        self.one_class_svm = ensemble_data["one_class_svm"]
        self.weights = ensemble_data["weights"]
        self.contamination = ensemble_data["contamination"]
        self.is_fitted = ensemble_data["is_fitted"]
        # Load adaptive threshold settings (with backward compatibility)
        self.use_adaptive_threshold = ensemble_data.get("use_adaptive_threshold", False)
        self.adaptive_threshold = ensemble_data.get("adaptive_threshold", None)

        # Load autoencoder if it exists
        autoencoder_path = model_path.replace(".pkl", "_autoencoder.h5")
        try:
            if len(self.weights) > 2 and self.weights[2] > 0:
                self.autoencoder.autoencoder = tf.keras.models.load_model(
                    autoencoder_path
                )
                self.autoencoder.is_fitted = True
        except Exception as e:
            self.logger.warning(f"Could not load autoencoder: {e}")
            if len(self.weights) > 2:
                self.weights[2] = 0  # Disable autoencoder

        self.logger.info(f"Ensemble model loaded from {model_path}")
        return self

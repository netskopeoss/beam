"""Trainer Module for creating custom app models"""

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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from .detect import (
    MultiHotEncoder,
    app_arr_non_numeric_feature_fields,
    app_meta_fields,
    app_numeric_feature_fields,
    app_str_non_numeric_feature_fields,
)
from .ensemble import EnsembleAnomalyDetector
from .utils import load_json_file, safe_create_path


class ModelTrainer:
    """Class for training custom app models"""

    def __init__(self, n_features: int = 50, min_transactions: int = 50):
        """
        Initialize the ModelTrainer.

        Args:
            n_features (int): The number of features to select for the model.
            min_transactions (int): Minimum transactions required for an app to be included.
        """
        self.n_features = n_features
        self.min_transactions = min_transactions
        self.logger = logging.getLogger(__name__)

        # Define transformers for feature processing
        # Note: We'll dynamically filter these based on available columns at training time
        self.numeric_feature_fields = app_numeric_feature_fields
        self.array_feature_fields = app_arr_non_numeric_feature_fields

    def get_available_transformers(self, X: pd.DataFrame) -> List[Tuple]:
        """Get transformers for columns that actually exist in the data."""
        transformers = []

        # Filter feature fields to only include columns that actually exist in the data
        available_numeric_fields = [
            col for col in self.numeric_feature_fields if col in X.columns
        ]
        available_array_fields = [
            col for col in self.array_feature_fields if col in X.columns
        ]

        # Create transformers only for columns that exist
        if available_numeric_fields:
            transformers.append(
                ("min_max_scaler", MinMaxScaler(), available_numeric_fields)
            )
        if available_array_fields:
            transformers.append(
                ("multi_hot_encoder", MultiHotEncoder(), available_array_fields)
            )

        return transformers

    def get_pipeline_estimator(
        self, n_estimators: int = 100, feature_count: int = None, X: pd.DataFrame = None
    ) -> Pipeline:
        """
        Create a pipeline estimator for model training.

        Args:
            n_estimators (int): Number of estimators for RandomForest and XGBoost.
            feature_count (int): Number of features available in the dataset.
                                If provided, max_features will be limited to this value.

        Returns:
            Pipeline: The configured pipeline estimator.
        """
        # Calculate max_features - should not exceed the available features
        if feature_count is not None:
            # If feature_count is explicitly provided, use it directly
            max_features = feature_count
        else:
            # Otherwise use the default n_features
            max_features = self.n_features

        # Use XGBoost model for feature selection instead of Random Forest
        feat_selure_selector = SelectFromModel(
            estimator=xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
                n_estimators=n_estimators,
                max_depth=4,
                learning_rate=0.1,
            ),
            threshold=-np.inf,  # Select based on max_features
            max_features=max_features,
        )

        xgb_classifier = xgb.XGBClassifier(
            objective="binary:logistic",  # for binary classification
            eval_metric="logloss",  # evaluation metric for binary classification
            random_state=42,
            n_jobs=-1,
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.1,
        )

        # Define the steps for the pipeline
        # Get transformers for available columns
        if X is not None:
            transformers = self.get_available_transformers(X)
        else:
            # Fallback to all transformers (for backward compatibility)
            transformers = [
                ("min_max_scaler", MinMaxScaler(), self.numeric_feature_fields),
                ("multi_hot_encoder", MultiHotEncoder(), self.array_feature_fields),
            ]

        steps = [
            (
                "ct",
                ColumnTransformer(transformers=transformers, remainder="drop"),
            ),
            ("feat_sel", feat_selure_selector),  # Use XGBoost for feature selection
            ("xgb", xgb_classifier),
        ]

        self.logger.info(
            "Creating pipeline with ColumnTransformer, XGBoost Feature Selector, XGBoost Classifier"
        )
        return Pipeline(steps=steps)

    def get_feature_names(
        self, ct: ColumnTransformer, char_limit: int = 500
    ) -> List[str]:
        """
        Get feature names from a fitted ColumnTransformer.

        Args:
            ct (ColumnTransformer): The fitted column transformer.
            char_limit (int): Maximum character length for feature names.

        Returns:
            List[str]: List of feature names.
        """
        feature_names = []

        # Get the actual columns used by each transformer
        for name, transformer, columns in ct.transformers_:
            if name == "min_max_scaler":
                for c in columns:
                    feature_names.append(c[:char_limit])
            elif name == "multi_hot_encoder":
                for i, c in enumerate(columns):
                    if hasattr(transformer, "classes_") and i < len(
                        transformer.classes_
                    ):
                        for p in transformer.classes_[i]:
                            feature_names.append(f"{c}_{p}"[:char_limit])
            elif name == "remainder":
                # Skip remainder columns
                pass

        return feature_names

    def convert_features_to_pd(
        self, features: List[Dict[str, Any]], target_label_name: str = "application"
    ) -> Tuple[pd.DataFrame, Any]:
        """
        Convert features to pandas DataFrame for model training.

        Args:
            features (List[Dict]): List of feature dictionaries.
            target_label_name (str): Name of the target label column.

        Returns:
            Tuple[pd.DataFrame, np.ndarray]: Features DataFrame and target labels.
        """
        # All feature fields
        feature_fields = (
            app_meta_fields
            + app_numeric_feature_fields
            + app_str_non_numeric_feature_fields
            + app_arr_non_numeric_feature_fields
        )

        # Convert to DataFrame
        features_pd = pd.DataFrame(features)

        # Get the target labels
        target_label = features_pd[target_label_name].values

        # Select only the feature columns needed for training
        feature_start_index = len(app_meta_fields)
        # Only select columns that actually exist in the DataFrame
        available_feature_fields = [
            col
            for col in feature_fields[feature_start_index:]
            if col in features_pd.columns
        ]
        features_train = features_pd[available_feature_fields]
        features_train = features_train.fillna(0)

        return features_train, target_label

    def train_model(
        self,
        training_data: List[Dict[str, Any]],
        app_name: str,
        all_domains: set = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Train an anomaly detection model for a specific app using ensemble methods.

        This method learns the normal behavior patterns of an application and creates
        an anomaly detector that can identify deviations from these patterns, which
        could indicate supply chain compromises.

        IMPORTANT: The training data is assumed to be clean (no anomalies/compromises).
        The contamination rate is set to near-zero (0.01%) to ensure the model doesn't
        falsely flag training samples as anomalous.

        Args:
            training_data (List[Dict[str, Any]]): List of feature dictionaries representing normal app behavior.
            app_name (str): Name of the app to train the model for.

        Returns:
            Dict[str, Any]: Trained anomaly detection model information.
        """
        self.logger.info(
            "Training anomaly detection model for application: %s", app_name
        )

        # Add the app name to each training sample
        for item in training_data:
            item["application"] = app_name

        # Filter out samples with too few transactions
        training_data = [
            item
            for item in training_data
            if item.get("transactions", 0) >= self.min_transactions
        ]

        if not training_data:
            self.logger.error(
                "No training data with sufficient transactions for %s", app_name
            )
            return None

        # Convert to pandas DataFrame - no need for synthetic negatives!
        X, _ = self.convert_features_to_pd(training_data)

        # Get transformers for available columns
        transformers = self.get_available_transformers(X)

        if not transformers:
            self.logger.error("No valid feature columns found for transformation")
            return None

        ct = ColumnTransformer(transformers=transformers, remainder="drop")
        X_transformed = ct.fit_transform(X)

        # Log dataset information
        transformed_feature_count = X_transformed.shape[1]
        self.logger.info(
            f"Dataset has {X.shape[1]} raw features, which transform to {transformed_feature_count} features"
        )
        self.logger.info(
            f"Training on {len(training_data)} samples of normal {app_name} behavior"
        )

        # Initialize ensemble anomaly detector
        # Set contamination to minimum value since training data is assumed clean
        # Using 0.0001 (0.01%) instead of 0 to avoid numerical issues in some algorithms
        contamination_rate = 0.0001  # 0.01% - essentially zero contamination
        ensemble_detector = EnsembleAnomalyDetector(
            contamination=contamination_rate,
            isolation_forest_params={
                "n_estimators": 100,
                "contamination": contamination_rate,
                "random_state": 42,
                "n_jobs": -1,
            },
            one_class_svm_params={
                "nu": contamination_rate,
                "gamma": "scale",
                "kernel": "rbf",
            },
            autoencoder_params={
                "encoding_dim": min(32, X_transformed.shape[1] // 2),
                "contamination": 0.001,  # Very low contamination for training
            },
            use_adaptive_threshold=True,  # Use adaptive threshold to prevent false positives on training data
        )

        # Train the ensemble on normal behavior
        ensemble_detector.fit(X_transformed)

        # Get feature names
        feature_names = self.get_feature_names(ct=ct)

        # Collect domain volume information for volume-weighted anomaly scoring
        domain_volumes = {}
        if all_domains:
            # Get volumes from training data for domains that were actually trained on
            for item in training_data:
                domain = item.get("domain", "")
                transactions = item.get("transactions", 0)
                if domain:
                    domain_volumes[domain] = transactions
            domain_list = sorted(list(all_domains))
            self.logger.info(
                f"Training includes {len(all_domains)} total domains for {app_name}: {domain_list}"
            )
            self.logger.info(f"Domain volumes: {domain_volumes}")
        else:
            training_domains = set()
            for item in training_data:
                domain = item.get("domain", "")
                transactions = item.get("transactions", 0)
                if domain:
                    training_domains.add(domain)
                    domain_volumes[domain] = transactions
            domain_list = sorted(list(training_domains))
            self.logger.info(
                f"Training includes {len(training_domains)} domains for {app_name}: {domain_list}"
            )
            self.logger.info(f"Domain volumes: {domain_volumes}")

        # Create model information dictionary
        model_info = {
            "key": app_name,
            "model_type": "ensemble_anomaly",
            "ensemble_detector": ensemble_detector,
            "feature_transformer": ct,
            "features": feature_names,
            "n_training_samples": len(training_data),
            "contamination": contamination_rate,
            "domain_volumes": domain_volumes,  # Store for volume-weighted scoring
        }

        self.logger.info(
            "Anomaly detection model training completed for %s with %d training samples",
            app_name,
            len(training_data),
        )

        return model_info

    def train_ensemble_model(
        self,
        training_data: List[Dict[str, Any]],
        app_name: str,
        use_ensemble: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Train an ensemble anomaly detection model for supply chain compromise detection.

        Args:
            training_data (List[Dict[str, Any]]): List of feature dictionaries for training.
            app_name (str): Name of the app to train the model for.
            use_ensemble (bool): Whether to use ensemble methods or fall back to XGBoost.

        Returns:
            Dict[str, Any]: Trained model information.
        """
        self.logger.info(
            "Training ensemble anomaly detection model for application: %s", app_name
        )

        # Add the app name to each training sample
        for item in training_data:
            item["application"] = app_name

        # Filter out samples with too few transactions
        training_data = [
            item
            for item in training_data
            if item.get("transactions", 0) >= self.min_transactions
        ]

        if not training_data:
            self.logger.error(
                "No training data with sufficient transactions for %s", app_name
            )
            return None

        # Convert to pandas DataFrame
        X, _ = self.convert_features_to_pd(training_data)

        # For ensemble anomaly detection, we assume training data is mostly normal
        # No need for synthetic negative examples - unsupervised approach

        if (
            use_ensemble and len(training_data) >= 10
        ):  # Need sufficient data for ensemble
            self.logger.info(
                f"Training ensemble model with {len(training_data)} normal samples"
            )

            # Transform features using the same pipeline approach
            transformers = self.get_available_transformers(X)
            ct = ColumnTransformer(transformers=transformers, remainder="passthrough")

            X_transformed = ct.fit_transform(X)

            # Initialize ensemble detector
            ensemble_detector = EnsembleAnomalyDetector(
                contamination=0.1,  # Expect 10% anomalies in future data
                isolation_forest_params={
                    "n_estimators": 100,
                    "contamination": 0.1,
                    "random_state": 42,
                    "n_jobs": -1,
                },
                one_class_svm_params={"nu": 0.1, "gamma": "scale", "kernel": "rbf"},
                autoencoder_params={
                    "encoding_dim": min(32, X_transformed.shape[1] // 2),
                    "contamination": 0.01,  # Reduced from 0.1 to 0.01 for less sensitivity
                },
            )

            # Train the ensemble
            ensemble_detector.fit(X_transformed)

            # Get feature names
            feature_names = self.get_feature_names(ct=ct)

            # Create model information dictionary
            model_info = {
                "key": app_name,
                "model_type": "ensemble",
                "estimator": ensemble_detector,
                "feature_transformer": ct,
                "features": feature_names,
                "n_training_samples": len(training_data),
            }

            self.logger.info(
                "Ensemble model training completed for %s with %d features",
                app_name,
                len(feature_names),
            )

        else:
            # Fall back to traditional supervised approach
            self.logger.info(
                f"Insufficient data for ensemble ({len(training_data)} samples), falling back to XGBoost"
            )
            model_info = self.train_model(training_data, app_name)
            if model_info:
                model_info["model_type"] = "xgboost"

        return model_info

    def save_model(self, model_info: Dict[str, Any], output_path: str) -> None:
        """
        Save a trained model to disk.

        Args:
            model_info (Dict[str, Any]): The model information to save.
            output_path (str): Path where the model will be saved.

        Returns:
            None
        """
        self.logger.info("Saving model to %s", output_path)

        # Ensure directory exists
        safe_create_path(output_path)

        # Save the model
        with open(output_path, "wb") as model_file:
            pickle.dump([model_info], model_file)

        self.logger.info("Model saved successfully to %s", output_path)

    def add_app_model(
        self, input_path: str, app_name: str, model_output_path: str
    ) -> Optional[str]:
        """
        Process input data to create and save a new app model.

        Args:
            input_path (str): Path to the input JSON file containing app features.
            app_name (str): Name of the app to train for.
            model_output_path (str): Path to save the model.

        Returns:
            Optional[str]: Path to the saved model file, or None if training failed.
        """
        self.logger.info(
            "Creating app model for %s using data from %s", app_name, input_path
        )

        # Load training data
        all_feature_data = load_json_file(input_path)

        # Ensure all_feature_data is a list
        if isinstance(all_feature_data, dict):
            all_feature_data = [all_feature_data]
        elif not isinstance(all_feature_data, list):
            raise ValueError(
                f"Training data must be a list or dict, got {type(all_feature_data)}"
            )

        # Extract ALL domains for this app (including those filtered out during training)
        all_domains_for_app = set()
        training_data = []

        for item in all_feature_data:
            if item.get("application") == app_name:
                domain = item.get("domain", "")
                if domain:
                    all_domains_for_app.add(domain)
                # Only include in training_data if it has sufficient transactions
                if item.get("transactions", 0) >= self.min_transactions:
                    training_data.append(item)

        self.logger.info(
            f"Found {len(all_domains_for_app)} total domains for {app_name}, {len(training_data)} meet training criteria"
        )

        # Train model
        model_info = self.train_model(training_data, app_name, all_domains_for_app)

        if model_info:
            # Save model
            self.save_model(model_info, model_output_path)
            self.logger.info("App model for %s created successfully", app_name)
            return model_output_path
        else:
            self.logger.error("Failed to create app model for %s", app_name)
            return None

    def extract_features_for_training(
        self, events_data: List[Dict[str, Any]], app_name: str
    ) -> pd.DataFrame:
        """
        Extract features from enriched events data for training.

        Args:
            events_data (List[Dict[str, Any]]): List of enriched event dictionaries.
            app_name (str): Name of the app to extract features for.

        Returns:
            pd.DataFrame: DataFrame containing extracted features.
        """
        self.logger.info("Extracting features for training app: %s", app_name)

        import os
        import tempfile

        from .features import aggregate_app_traffic

        # Create temporary files for processing
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as temp_input:
            temp_input_path = temp_input.name
            # Save events data to temporary file
            import json

            json.dump(events_data, temp_input)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as temp_output:
            temp_output_path = temp_output.name

        try:
            # Extract features using the existing aggregate_app_traffic function
            aggregate_app_traffic(
                fields=["useragent", "domain"],
                input_path=temp_input_path,
                output_path=temp_output_path,
                min_transactions=self.min_transactions,
            )

            # Load the extracted features
            features_data = load_json_file(temp_output_path)

            if not features_data:
                self.logger.warning("No features extracted for %s", app_name)
                return pd.DataFrame()

            # Ensure it's a list
            if isinstance(features_data, dict):
                features_data = [features_data]

            # Convert to DataFrame
            features_df = pd.DataFrame(features_data)
            self.logger.info(
                "Extracted %d feature records for %s", len(features_df), app_name
            )

            return features_df

        finally:
            # Clean up temporary files
            try:
                os.unlink(temp_input_path)
                os.unlink(temp_output_path)
            except OSError:
                pass


def extract_app_features(
    input_data_path: str,
    output_path: str,
    min_transactions: int = 50,
    fields: Optional[List[str]] = None,
) -> str:
    """
    Extract application features from enriched events.

    Args:
        input_data_path (str): Path to the input data file (enriched events).
        output_path (str): Path to save the extracted features.
        min_transactions (int): Minimum number of transactions required.
        fields (List[str]): Fields to use as identifiers for aggregation.

    Returns:
        str: Path to the saved features file.
    """
    if fields is None:
        fields = ["useragent", "domain"]

    logger = logging.getLogger(__name__)
    logger.info("Extracting app features from %s", input_data_path)

    from .features import aggregate_app_traffic

    aggregate_app_traffic(
        fields=fields,
        input_path=input_data_path,
        output_path=output_path,
        min_transactions=min_transactions,
    )

    logger.info("Features extracted and saved to %s", output_path)
    return output_path


def train_custom_app_model(
    features_path: str,
    app_name: str,
    output_model_path: str,
    n_features: int = 50,
    min_transactions: int = 50,
) -> str:
    """
    Train a custom application model from extracted features.

    Args:
        features_path (str): Path to the features file.
        app_name (str): Name of the application to train for.
        output_model_path (str): Path to save the trained model.
        n_features (int): Number of features to select.
        min_transactions (int): Minimum transactions required.

    Returns:
        str: Path to the saved model file.
    """
    logger = logging.getLogger(__name__)
    logger.info("Training custom model for %s", app_name)

    # Create trainer
    trainer = ModelTrainer(n_features=n_features, min_transactions=min_transactions)

    # Train and save model
    model_path = trainer.add_app_model(
        input_path=features_path, app_name=app_name, model_output_path=output_model_path
    )

    if model_path:
        logger.info("Custom model for %s trained and saved to %s", app_name, model_path)
    return model_path

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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from .detect import MultiHotEncoder
from .features import (
    app_arr_non_numeric_feature_fields,
    app_meta_fields,
    app_numeric_feature_fields,
    app_str_non_numeric_feature_fields,
)
from .utils import load_json_file, safe_create_path, save_json_data


class ModelTrainer:
    """Class for training custom app models"""

    def __init__(self, n_features: int = 150, min_transactions: int = 50):
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
        self.columns_to_be_transformed = [
            ("min_max_scaler", MinMaxScaler(), app_numeric_feature_fields),
            (
                "multi_hot_encoder",
                MultiHotEncoder(),
                app_arr_non_numeric_feature_fields,
            ),
        ]

    def get_pipeline_estimator(self, n_estimators: int = 100) -> Pipeline:
        """
        Create a pipeline estimator for model training.

        Args:
            n_estimators (int): Number of estimators for RandomForest and XGBoost.

        Returns:
            Pipeline: The configured pipeline estimator.
        """
        # Feature selector based on Random Forest importance
        rf_feature_selector = SelectFromModel(
            estimator=RandomForestClassifier(
                n_estimators=n_estimators,
                criterion="entropy",
                random_state=42,
                n_jobs=-1,
            ),
            threshold=-np.inf,  # Select based on max_features
            max_features=self.n_features,
        )

        xgb_classifier = xgb.XGBClassifier(
            objective="multi:softprob",  # for multiclass classification
            eval_metric="mlogloss",  # common evaluation metric for multiclass
            random_state=42,
            n_jobs=-1,
            n_estimators=n_estimators,
            max_depth=15,
            learning_rate=0.1,
        )

        # Define the steps for the pipeline
        steps = [
            (
                "ct",
                ColumnTransformer(
                    transformers=self.columns_to_be_transformed, remainder="drop"
                ),
            ),
            ("rf_feat", rf_feature_selector),
            ("xgb", xgb_classifier),
        ]

        self.logger.info(
            "Creating pipeline with ColumnTransformer, RF Selector, XGBoost Classifier"
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

        for c in app_numeric_feature_fields:
            feature_names.append(c[:char_limit])

        for i, c in enumerate(app_arr_non_numeric_feature_fields):
            for p in ct.named_transformers_["multi_hot_encoder"].classes_[i]:
                feature_names.append(f"{c}_{p}"[:char_limit])

        return feature_names

    def convert_features_to_pd(
        self, features: List[Dict], target_label_name: str = "application"
    ) -> Tuple[pd.DataFrame, np.ndarray]:
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
        features_train = features_pd[feature_fields[feature_start_index:]]
        features_train = features_train.fillna(0)

        return features_train, target_label

    def train_model(self, training_data: List[Dict], app_name: str) -> Dict[str, Any]:
        """
        Train a model for a specific app.

        Args:
            training_data (List[Dict]): List of feature dictionaries for training.
            app_name (str): Name of the app to train the model for.

        Returns:
            Dict[str, Any]: Trained model information.
        """
        self.logger.info(f"Training model for application: {app_name}")

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
                f"No training data with sufficient transactions for {app_name}"
            )
            return None

        # Convert to pandas DataFrame
        X, y = self.convert_features_to_pd(training_data)

        # Create and fit the model
        estimator = self.get_pipeline_estimator()
        estimator.fit(X, y)

        # Get feature names
        feature_names = self.get_feature_names(ct=estimator.named_steps["ct"])
        selected_feat_ind = estimator.named_steps["rf_feat"].get_support()
        selected_features = np.array(feature_names)[selected_feat_ind]

        # Create model information dictionary
        model_info = {
            "key": app_name,
            "estimator": estimator,
            "features": feature_names,
            "selected_features": selected_features,
        }

        self.logger.info(
            f"Model training completed for {app_name} with {len(selected_features)} selected features"
        )

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
        self.logger.info(f"Saving model to {output_path}")

        # Ensure directory exists
        safe_create_path(str(Path(output_path).parent))

        # Save the model
        with open(output_path, "wb") as model_file:
            pickle.dump([model_info], model_file)

        self.logger.info(f"Model saved successfully to {output_path}")

    def add_app_model(
        self, input_path: str, app_name: str, model_output_path: str
    ) -> None:
        """
        Process input data to create and save a new app model.

        Args:
            input_path (str): Path to the input JSON file containing app features.
            app_name (str): Name of the app to train for.
            model_output_path (str): Path to save the model.

        Returns:
            None
        """
        self.logger.info(
            f"Creating app model for {app_name} using data from {input_path}"
        )

        # Load training data
        training_data = load_json_file(input_path)

        # Train model
        model_info = self.train_model(training_data, app_name)

        if model_info:
            # Save model
            self.save_model(model_info, model_output_path)
            self.logger.info(f"App model for {app_name} created successfully")
        else:
            self.logger.error(f"Failed to create app model for {app_name}")

    def merge_models(
        self, existing_model_path: str, new_model_path: str, output_path: str
    ) -> None:
        """
        Merge a newly created model with an existing model file.

        Args:
            existing_model_path (str): Path to the existing model file.
            new_model_path (str): Path to the new model file.
            output_path (str): Path to save the merged model.

        Returns:
            None
        """
        self.logger.info(
            f"Merging models from {existing_model_path} and {new_model_path}"
        )

        try:
            # Load existing models
            with open(existing_model_path, "rb") as existing_file:
                existing_models = pickle.load(existing_file)

            # Load new model
            with open(new_model_path, "rb") as new_file:
                new_model = pickle.load(new_file)

            # Combine models
            combined_models = existing_models + new_model

            # Save merged models
            with open(output_path, "wb") as output_file:
                pickle.dump(combined_models, output_file)

            self.logger.info(f"Models merged successfully and saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Error merging models: {str(e)}")


def extract_app_features(
    input_data_path: str,
    output_path: str,
    min_transactions: int = 50,
    fields: List[str] = ["useragent"],
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
    logger = logging.getLogger(__name__)
    logger.info(f"Extracting app features from {input_data_path}")

    from .features import aggregate_app_traffic

    aggregate_app_traffic(
        fields=fields,
        input_path=input_data_path,
        output_path=output_path,
        min_transactions=min_transactions,
    )

    logger.info(f"Features extracted and saved to {output_path}")
    return output_path


def train_custom_app_model(
    features_path: str,
    app_name: str,
    output_model_path: str,
    n_features: int = 150,
    min_transactions: int = 50,
) -> None:
    """
    Train a custom application model from extracted features.

    Args:
        features_path (str): Path to the features file.
        app_name (str): Name of the application to train for.
        output_model_path (str): Path to save the trained model.
        n_features (int): Number of features to select.
        min_transactions (int): Minimum transactions required.

    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Training custom model for {app_name}")

    # Create trainer
    trainer = ModelTrainer(n_features=n_features, min_transactions=min_transactions)

    # Train and save model
    trainer.add_app_model(
        input_path=features_path, app_name=app_name, model_output_path=output_model_path
    )

    logger.info(f"Custom model for {app_name} trained and saved to {output_model_path}")

    logger.info(f"Custom model for {app_name} trained and saved to {output_model_path}")

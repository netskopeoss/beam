"""Detect Module"""

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
from typing import Any, Dict, Set, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from path import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

from .utils import load_json_file, safe_create_path, save_json_data

app_meta_fields = ["key", "application"]

app_numeric_feature_fields = [
    "transactions",
    "refered_traffic_pct",
    "referer_domain_cnt",
    "unique_actions",
    "http_status_cnt",
    "http_method_cnt",
    "req_content_type_cnt",
    "resp_content_type_cnt",
    "avg_time_interval_sec",
    "std_time_interval_sec",
    "median_time_interval_sec",
    "range_time_interval_sec",
    "range_timestamp",
    "max_time_taken_ms",
    "min_time_taken_ms",
    "sum_time_taken_ms",
    "avg_time_taken_ms",
    "std_time_taken_ms",
    "median_time_taken_ms",
    "range_time_taken_ms",
    "max_client_bytes",
    "min_client_bytes",
    "sum_client_bytes",
    "avg_client_bytes",
    "std_client_bytes",
    "median_client_bytes",
    "range_client_bytes",
    "max_server_bytes",
    "min_server_bytes",
    "sum_server_bytes",
    "avg_server_bytes",
    "std_server_bytes",
    "median_server_bytes",
    "range_server_bytes",
    "web_traffic_pct",
    "cloud_traffic_pct",
    "sequence_num_keys",
    "sequence_max_key_length",
    "sequence_min_key_length",
    "sequence_max_val",
    "sequence_min_val",
    "sequence_sum_val",
    "sequence_avg_val",
    "sequence_std_val",
    "sequence_median_val",
    "sequence_range_val",
]

app_str_non_numeric_feature_fields = ["domain"]

app_arr_non_numeric_feature_fields = [
    "http_methods",
    "http_statuses",
    "req_content_types",
    "resp_content_types",
]

app_feature_fields = (
    app_meta_fields
    + app_numeric_feature_fields
    + app_str_non_numeric_feature_fields
    + app_arr_non_numeric_feature_fields
)


malware_meta_fields = ["key"]
malware_feature_columns = malware_meta_fields + [
    "referer_domain_cnt",
    "unique_actions",
    "key_hostname_cnt",
    "domain_cnt",
    "http_status_cnt",
    "http_method_cnt",
    "req_content_type_cnt",
    "resp_content_type_cnt",
    "avg_time_interval_sec",
    "std_time_interval_sec",
    "median_time_interval_sec",
    "range_time_interval_sec",
    "range_timestamp",
    "max_time_taken_ms",
    "min_time_taken_ms",
    "sum_time_taken_ms",
    "avg_time_taken_ms",
    "std_time_taken_ms",
    "median_time_taken_ms",
    "range_time_taken_ms",
    "max_client_bytes",
    "min_client_bytes",
    "sum_client_bytes",
    "avg_client_bytes",
    "std_client_bytes",
    "median_client_bytes",
    "range_client_bytes",
    "max_server_bytes",
    "min_server_bytes",
    "sum_server_bytes",
    "avg_server_bytes",
    "std_server_bytes",
    "median_server_bytes",
    "range_server_bytes",
    "refered_traffic_pct",
    "web_traffic_pct",
    "cloud_traffic_pct",
    "sequence_num_keys",
    "sequence_max_key_length",
    "sequence_min_key_length",
    "sequence_max_val",
    "sequence_min_val",
    "sequence_sum_val",
    "sequence_avg_val",
    "sequence_std_val",
    "sequence_median_val",
    "sequence_range_val",
    "key_hostnames",
    "http_methods",
    "http_statuses",
    "req_content_types",
    "resp_content_types",
]


def load_app_model(app_model_path: str) -> Any:
    """
    Load the application model from the specified path.

    Args:
        app_model_path (str): The path to the app model file.

    Returns:
        model: The loaded model.
    """
    with open(app_model_path, "rb") as _file:
        model = pickle.load(_file)[0]
    return model


def load_domain_model(domain_model_path: Path) -> Tuple[Set, Dict]:
    """
    Load the domain model from the specified path.

    Args:
        domain_model_path (str): The path to the domain model file.

    Returns:
        Tuple[set, dict]: A set of apps and a dictionary of models.
    """
    try:
        with open(domain_model_path, "rb") as _file:
            # Suppress sklearn warnings during model loading
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=DeprecationWarning, message='.*__sklearn_tags__.*')
                raw_models = pickle.load(_file)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to load domain model {domain_model_path}: {e}")
        logger.error("This may be due to version incompatibility. Try retraining the model.")
        return set(), dict()

    models = dict()
    apps = set()

    for raw_model in raw_models:
        app = raw_model.pop("key")
        models[app] = raw_model
        apps.add(app)

    return apps, models


class MultiHotEncoder(BaseEstimator, TransformerMixin):
    """Wraps `MultiLabelBinarizer` in a form that can work with `ColumnTransformer`. Note
    that input X has to be a `pandas.DataFrame`.
    """

    def __init__(self) -> None:
        self.mlbs = []
        self.n_columns = 0
        self.categories_ = self.classes_ = []

    def fit(self, X: pd.DataFrame, y=None) -> BaseEstimator:
        """
        Fit the MultiHotEncoder to the data.

        Args:
            X (pd.DataFrame): The input data to fit.
            y: Ignored, not used in this transformer.

        Returns:
            self: The fitted transformer.
        """
        for i in range(X.shape[1]):  # X can be of multiple columns
            mlb = MultiLabelBinarizer()
            mlb.fit(X.iloc[:, i])
            self.mlbs.append(mlb)
            self.classes_.append(mlb.classes_)
            self.n_columns += 1
        return self

    def transform(self, X: pd.DataFrame) -> NDArray:
        """
        Transform the input data using the fitted MultiHotEncoder.

        Args:
            X (pd.DataFrame): The input data to transform.

        Returns:
            np.ndarray: The transformed data as a numpy array.

        Raises:
            ValueError: If the transformer has not been fitted or if the number
                        of columns in the input data does not match the fitted data.
        """
        if self.n_columns == 0:
            raise ValueError("Please fit the transformer first.")
        if self.n_columns != X.shape[1]:
            raise ValueError(
                f"The fit transformer deals with {self.n_columns} columns "
                f"while the input has {X.shape[1]}."
            )
        result = list()
        for i in range(self.n_columns):
            result.append(self.mlbs[i].transform(X.iloc[:, i]))

        result = np.concatenate(result, axis=1)
        return result


def convert_summary_to_features(input_data: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert the input summary data to features.

    Args:
        input_data (dict): The input summary data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The original features and the processed features.
    """
    features_og = pd.json_normalize(input_data)
    features_og.reset_index()
    features_og = features_og[malware_feature_columns]

    feature_start_index = len(malware_meta_fields)
    features_pd = pd.DataFrame(
        data=features_og.to_numpy()[:, feature_start_index:],
        columns=features_og.columns[feature_start_index:],
    )
    features_pd.reset_index()
    return features_og, features_pd


def convert_supply_chain_summaries_to_features(
    input_data: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert the input supply chain summaries to features.

    Args:
        input_data (dict): The input supply chain summaries.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The original features and the processed features.
    """
    features_og = pd.json_normalize(input_data)
    features_og.reset_index()
    features_og = features_og[app_feature_fields]

    feature_start_index = len(app_meta_fields)
    features_pd = pd.DataFrame(
        data=features_og.to_numpy()[:, feature_start_index:],
        columns=features_og.columns[feature_start_index:],
    )
    features_pd.reset_index()
    return features_og, features_pd


def detect_anomalous_domain(
    input_path: str,
    domain_model_path: Path,
    app_prediction_dir: str,
    prob_cutoff: float = 0.8,
) -> None:
    """
    Detect anomalous domains in the given input data.

    Args:
        input_path (str): The path to the input JSON file containing the data.
        app_model_path (str): The path to the app model file.
        app_prediction_dir (str): The directory to save the predictions.
        prob_cutoff (float): The cutoff for probability to determine if it's anomalous.

    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    apps, models = load_domain_model(domain_model_path)
    logger.info("[x] Apps found in the model: " + str(apps))
    logger.info("[x] Loading traffic from: " + str(input_path))
    features_og, features_pd = convert_supply_chain_summaries_to_features(
        load_json_file(input_path)
    )

    for observation_index, observation_series in features_og.iterrows():
        application = observation_series["application"]
        if application not in models:
            logger.info(
                "[x] Application not found in domain models: " + str(application)
            )
        else:
            logger.info(
                "[x] Application found to test supply chain compromises against: "
                + str(application)
            )
            observation_key = observation_series["key"]
            model = models[application]

            estimator = model["estimator"]
            selected_feature_names = model["selected_features"]
            
            # Get classes from the final classifier in the pipeline
            try:
                if hasattr(estimator, "named_steps") and "xgb_classifier" in estimator.named_steps:
                    classes = estimator.named_steps["xgb_classifier"].classes_
                elif hasattr(estimator, "classes_"):
                    classes = estimator.classes_
                else:
                    # Fallback - try to get classes from the last step of the pipeline
                    classes = estimator.steps[-1][1].classes_
            except AttributeError as e:
                logger.error(f"Failed to get classes from estimator: {e}")
                continue

            # Make predictions with error handling for sklearn compatibility issues
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=DeprecationWarning)
                    predictions = estimator.predict_proba(features_pd)
            except AttributeError as e:
                if "__sklearn_tags__" in str(e):
                    logger.error(f"sklearn compatibility issue with model for {application}: {e}")
                    logger.error("This indicates a version compatibility problem. The model may need to be retrained with compatible sklearn/xgboost versions.")
                    continue
                else:
                    raise

            # Try to get features using XGBoost feature selector first, then fall back to RF if needed
            if (
                hasattr(estimator, "named_steps")
                and "xgb_feat" in estimator.named_steps
            ):
                features_scaled = estimator["xgb_feat"].transform(
                    estimator["ct"].transform(features_pd)
                )
            elif (
                hasattr(estimator, "named_steps") and "rf_feat" in estimator.named_steps
            ):
                features_scaled = estimator["rf_feat"].transform(
                    estimator["ct"].transform(features_pd)
                )
            else:
                # Try accessing as dictionary for backward compatibility
                if "xgb_feat" in estimator:
                    features_scaled = estimator["xgb_feat"].transform(
                        estimator["ct"].transform(features_pd)
                    )
                else:
                    features_scaled = estimator["rf_feat"].transform(
                        estimator["ct"].transform(features_pd)
                    )

            predicted_class_index = predictions[observation_index].argmax()
            predicted_class_name = classes[predicted_class_index]
            predicted_class_proba = predictions[
                observation_index, predicted_class_index
            ]

            obs_file_dir = (
                str(observation_index)
                + "_"
                + observation_key.replace(" ", "_")
                .replace("'", "")
                .replace("/", "")[:35]
            )
            parent_dir = f"{app_prediction_dir}/{obs_file_dir}/"
            safe_create_path(parent_dir)

            full_predictions = sorted(
                [
                    {"class": str(c), "probability": round(float(100.0 * p), 4)}
                    for p, c in zip(predictions[observation_index], classes)
                ],
                key=lambda x: x["probability"],
                reverse=True,
            )
            full_predictions_path = f"{parent_dir}full_predictions.json"
            save_json_data(full_predictions, full_predictions_path)


def detect_anomalous_domain_with_custom_model(
    input_path: str,
    custom_model_path: Path,
    app_prediction_dir: str,
    prob_cutoff: float = 0.8,
) -> None:
    """
    Detect anomalous domains using an individual custom model.

    Args:
        input_path (str): The path to the input JSON file containing the data.
        custom_model_path (Path): The path to the custom model file.
        app_prediction_dir (str): The directory to save the predictions.
        prob_cutoff (float): The cutoff for probability to determine if it's anomalous.

    Returns:
        None
    """
    logger = logging.getLogger(__name__)

    # Load the individual custom model
    try:
        with open(custom_model_path, "rb") as _file:
            # Suppress sklearn warnings during model loading
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=DeprecationWarning, message='.*__sklearn_tags__.*')
                raw_models = pickle.load(_file)
    except Exception as e:
        logger.error(f"Failed to load custom model {custom_model_path}: {e}")
        logger.error("This may be due to version incompatibility. Try retraining the model.")
        return

    # Convert single model to the expected format
    models = dict()
    apps = set()

    if isinstance(raw_models, list) and len(raw_models) > 0:
        raw_model = raw_models[0].copy()  # Take the first (and likely only) model
        app = raw_model.pop("key")
        models[app] = raw_model
        apps.add(app)
    else:
        logger.error(f"Unexpected model format in {custom_model_path}")
        return

    features_og, features_pd = convert_supply_chain_summaries_to_features(
        load_json_file(input_path)
    )

    for observation_index, observation_series in features_og.iterrows():
        application = observation_series["application"]
        if application not in models:
            logger.info(
                "[x] Application not found in custom models: " + str(application)
            )
        else:
            logger.info(
                "[x] Application found to test supply chain compromises against: "
                + str(application)
            )
            observation_key = observation_series["key"]
            model = models[application]

            estimator = model["estimator"]
            selected_feature_names = model["selected_features"]
            
            # Get classes from the final classifier in the pipeline
            try:
                if hasattr(estimator, "named_steps") and "xgb_classifier" in estimator.named_steps:
                    classes = estimator.named_steps["xgb_classifier"].classes_
                elif hasattr(estimator, "classes_"):
                    classes = estimator.classes_
                else:
                    # Fallback - try to get classes from the last step of the pipeline
                    classes = estimator.steps[-1][1].classes_
            except AttributeError as e:
                logger.error(f"Failed to get classes from estimator: {e}")
                continue

            # Make predictions with error handling for sklearn compatibility issues
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=DeprecationWarning)
                    predictions = estimator.predict_proba(features_pd)
            except AttributeError as e:
                if "__sklearn_tags__" in str(e):
                    logger.error(f"sklearn compatibility issue with model for {application}: {e}")
                    logger.error("This indicates a version compatibility problem. The model may need to be retrained with compatible sklearn/xgboost versions.")
                    continue
                else:
                    raise

            # Try to get features using XGBoost feature selector first, then fall back to RF if needed
            if (
                hasattr(estimator, "named_steps")
                and "xgb_feat" in estimator.named_steps
            ):
                features_scaled = estimator["xgb_feat"].transform(
                    estimator["ct"].transform(features_pd)
                )
            elif (
                hasattr(estimator, "named_steps") and "rf_feat" in estimator.named_steps
            ):
                features_scaled = estimator["rf_feat"].transform(
                    estimator["ct"].transform(features_pd)
                )
            else:
                # Try accessing as dictionary for backward compatibility
                if "xgb_feat" in estimator:
                    features_scaled = estimator["xgb_feat"].transform(
                        estimator["ct"].transform(features_pd)
                    )
                else:
                    features_scaled = estimator["rf_feat"].transform(
                        estimator["ct"].transform(features_pd)
                    )

            predicted_class_index = predictions[observation_index].argmax()
            predicted_class_name = classes[predicted_class_index]
            predicted_class_proba = predictions[
                observation_index, predicted_class_index
            ]

            obs_file_dir = (
                str(observation_index)
                + "_"
                + observation_key.replace(" ", "_")
                .replace("'", "")
                .replace("/", "")[:35]
            )
            parent_dir = f"{app_prediction_dir}/{obs_file_dir}/"
            safe_create_path(parent_dir)

            full_predictions = sorted(
                [
                    {"class": str(c), "probability": round(float(100.0 * p), 4)}
                    for p, c in zip(predictions[observation_index], classes)
                ],
                key=lambda x: x["probability"],
                reverse=True,
            )
            full_predictions_path = f"{parent_dir}full_predictions.json"
            save_json_data(full_predictions, full_predictions_path)

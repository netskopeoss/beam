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
import os
import pickle
from typing import Any, Dict, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from numpy.typing import NDArray
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


def load_domain_model(domain_model_path: str) -> Tuple[Set, Dict]:
    """
    Load the domain model from the specified path.

    Args:
        domain_model_path (str): The path to the domain model file.

    Returns:
        Tuple[set, dict]: A set of apps and a dictionary of models.
    """
    with open(domain_model_path, "rb") as _file:
        raw_models = pickle.load(_file)

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


def detect_anomalous_app(
    input_path: str,
    app_model_path: str,
    app_prediction_directory: str,
    custom_model_path: Optional[str] = None,
) -> None:
    """
    Detect anomalous applications in the given input data.

    Args:
        input_path (str): The path to the input JSON file containing the data.
        app_model_path (str): The path to the pickled model.
        app_prediction_directory (str): The directory to save the predictions.
        custom_model_path (str, optional): Path to a custom model to use alongside the standard model.

    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    logger.info(f"[x] Detecting anomalous applications in {input_path}")

    # Load the primary model
    model = load_app_model(app_model_path)

    # If a custom model is specified, try to load and use it
    if custom_model_path and os.path.exists(custom_model_path):
        logger.info(f"[x] Using custom model from: {custom_model_path}")
        model = load_app_model(custom_model_path)
    estimator = model["estimator"]
    selected_feature_names = model["selected_features"]
    classes = estimator.classes_

    features_og, features_pd = convert_summary_to_features(load_json_file(input_path))
    predictions = estimator.predict_proba(features_pd)
    features_scaled = estimator["rf_feat"].transform(
        estimator["ct"].transform(features_pd)
    )

    for observation_index, observation_key in enumerate(features_og["key"]):
        predicted_class_index = predictions[observation_index].argmax()
        predicted_class_name = classes[predicted_class_index]
        predicted_class_proba = predictions[observation_index, predicted_class_index]

        plt.clf()
        chosen_instance = features_scaled[observation_index, :]
        explainer = shap.TreeExplainer(estimator["rf"])
        shap_values = explainer.shap_values(chosen_instance)
        shap.initjs()
        exp = shap.Explanation(
            values=shap_values[..., predicted_class_index],
            base_values=explainer.expected_value[predicted_class_index],
            data=chosen_instance,
            feature_names=selected_feature_names,
        )

        shap.waterfall_plot(exp, max_display=20, show=False)
        obs_file_dir = (
            str(observation_index)
            + "_"
            + observation_key.replace(" ", "_").replace("'", "").replace("/", "")[:35]
        )
        parent_dir = f"{app_prediction_directory}/{obs_file_dir}/"
        safe_create_path(parent_dir)
        exp_png_path = f"{parent_dir}{predicted_class_name}_shap_waterfall.png"
        plt.savefig(exp_png_path, dpi=150, bbox_inches="tight")

        full_predictions = sorted(
            [
                {"class": c, "probability": round(100.0 * p, 4)}
                for p, c in zip(predictions[observation_index], classes)
            ],
            key=lambda x: x["probability"],
            reverse=True,
        )
        full_predictions_path = f"{parent_dir}full_predictions.json"
        save_json_data(full_predictions, full_predictions_path)

        logger.info(
            f"""
            i = {observation_index}
            Advertised user agent = {observation_key}
            Predicted class = {predicted_class_name} ({round(predicted_class_proba * 100, 2)}%)
            Top 3 predictions = {full_predictions[:3]}
            SHAP explanation path = {exp_png_path}
            Full predictions path = {full_predictions_path}
        \n"""
        )


def detect_anomalous_domain(
    input_path: str,
    domain_model_path: str,
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
            classes = estimator.classes_

            predictions = estimator.predict_proba(features_pd)
            features_scaled = estimator["rf_feat"].transform(
                estimator["ct"].transform(features_pd)
            )

            predicted_class_index = predictions[observation_index].argmax()
            predicted_class_name = classes[predicted_class_index]
            predicted_class_proba = predictions[
                observation_index, predicted_class_index
            ]

            plt.clf()

            try:
                chosen_instance = features_scaled[observation_index, :].toarray()
            except:
                chosen_instance = features_scaled[observation_index, :]

            explainer = shap.TreeExplainer(estimator["rf"])
            shap_values = explainer.shap_values(chosen_instance)
            shap.initjs()
            exp = shap.Explanation(
                values=shap_values[..., predicted_class_index],
                base_values=explainer.expected_value[predicted_class_index],
                data=chosen_instance,
                feature_names=selected_feature_names,
            )

            try:
                shap.waterfall_plot(exp[0], max_display=20, show=False)
            except:
                shap.waterfall_plot(exp, max_display=20, show=False)

            obs_file_dir = (
                str(observation_index)
                + "_"
                + observation_key.replace(" ", "_")
                .replace("'", "")
                .replace("/", "")[:35]
            )
            parent_dir = f"{app_prediction_dir}/{obs_file_dir}/"
            safe_create_path(parent_dir)
            exp_png_path = f"{parent_dir}{predicted_class_name}_shap_waterfall.png"
            plt.savefig(exp_png_path, dpi=150, bbox_inches="tight")

            full_predictions = sorted(
                [
                    {"class": c, "probability": round(100.0 * p, 4)}
                    for p, c in zip(predictions[observation_index], classes)
                ],
                key=lambda x: x["probability"],
                reverse=True,
            )
            full_predictions_path = f"{parent_dir}full_predictions.json"
            save_json_data(full_predictions, full_predictions_path)

            if (predicted_class_name == "Not specified app") and (
                predicted_class_proba >= prob_cutoff
            ):
                logger.info("[!!] Potential supply chain compromise found ")
                logger.info(
                    f"""
                    i = {observation_index}
                    {observation_key}
                    Predicted class = {predicted_class_name} ({round(predicted_class_proba * 100, 2)}%)
                    Top 3 predictions = {full_predictions[:3]}
                    Full predictions path = {full_predictions_path}
                \n"""
                )

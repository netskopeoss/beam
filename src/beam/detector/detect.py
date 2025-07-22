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
from .explainer import ModelExplainer
from beam import constants
from .models import (
    PredictionClass, 
    TopFeature, 
    ExplanationJson, 
    AnomalyInfo, 
    DetectionResults
)

app_meta_fields = ["key", "application"]

app_numeric_feature_fields = [
    "transactions", "refered_traffic_pct", "referer_domain_cnt", "unique_actions",

    "http_status_cnt", "http_method_cnt", "req_content_type_cnt", "resp_content_type_cnt", "range_timestamp",

    "max_time_taken_ms", "min_time_taken_ms", "sum_time_taken_ms", "avg_time_taken_ms", "std_time_taken_ms",
    "median_time_taken_ms", "range_time_taken_ms",
    "max_client_bytes", "min_client_bytes", "sum_client_bytes", "avg_client_bytes", "std_client_bytes",
    "median_client_bytes", "range_client_bytes",
    "max_server_bytes", "min_server_bytes", "sum_server_bytes", "avg_server_bytes", "std_server_bytes",
    "median_server_bytes", "range_server_bytes",

    'web_traffic_pct', 'cloud_traffic_pct',

    "sequence_num_keys", "sequence_max_key_length", "sequence_min_key_length", "sequence_max_val", "sequence_min_val",
    "sequence_sum_val", "sequence_avg_val", "sequence_std_val", "sequence_median_val", "sequence_range_val",

    'key_hostname_cnt', 'domain_cnt', 'avg_interval_sec', 'burst_ratio',
    'night_activity_ratio', 'hour_entropy', 'redirect_ratio', 'error_ratio', 'avg_path_depth', 'path_depth_variance',
    'response_size_cv', 'https_ratio', 'http2_usage_ratio', 'http_version_diversity', 'ua_diversity', 'bot_ratio',
    'suspicious_ua_ratio', 'same_origin_referer_ratio', 'cdn_usage_ratio', 'domain_concentration',
    'iqr_server_bytes', 'iqr_client_bytes', 'iqr_time_taken_ms', 'skewness_server_bytes', 'skewness_client_bytes',
    'skewness_time_taken_ms', 'kurtosis_server_bytes', 'kurtosis_client_bytes', 'kurtosis_time_taken_ms', 'url_entropy',
    'ua_entropy', 'domain_entropy', 'ua_consistency',

    'api_endpoint_ratio', 'automation_suspicion', 'avg_bytes_per_request', 'avg_compression_ratio', 'avg_domain_length',
    'avg_subdomain_depth', 'avg_time_interval_sec', 'avg_url_length', 'cert_chain_depth_estimate', 'chrome_ratio',
    'compression_usage_ratio', 'content_type_mismatch_ratio', 'cross_domain_ratio', 'css_ratio', 'cv_client_bytes',
    'cv_server_bytes', 'cv_time_interval_sec', 'cv_time_taken_ms', 'distinct_key_domain_count', 'error_rate',
    'executable_ratio',
    'external_domain_count', 'external_domain_ratio', 'firefox_ratio', 'html_ratio', 'html_response_ratio',
    'image_ratio',
    'internal_domain_count', 'interval_entropy', 'interval_regularity', 'iqr_time_interval_sec', 'js_ratio',
    'json_ratio',
    'json_response_ratio', 'kurtosis_time_interval_sec', 'large_response_ratio', 'mad_client_bytes',
    'mad_server_bytes', 'mad_time_interval_sec', 'mad_time_taken_ms', 'max_domain_length', 'max_subdomain_depth',
    'max_time_interval_sec', 'median_time_interval_sec', 'min_time_interval_sec', 'mixed_content_risk',
    'new_domain_ratio',
    'non_standard_method_ratio', 'numeric_domain_ratio', 'outlier_ratio_client_bytes', 'outlier_ratio_server_bytes',
    'outlier_ratio_time_interval_sec', 'outlier_ratio_time_taken_ms', 'p25_client_bytes', 'p25_server_bytes',
    'p25_time_interval_sec',
    'p25_time_taken_ms', 'p75_client_bytes', 'p75_server_bytes', 'p75_time_interval_sec', 'p75_time_taken_ms',
    'p90_client_bytes',
    'p90_server_bytes', 'p90_time_interval_sec', 'p90_time_taken_ms', 'p95_client_bytes', 'p95_server_bytes',
    'p95_time_interval_sec',
    'p95_time_taken_ms', 'p99_client_bytes', 'p99_server_bytes', 'p99_time_interval_sec', 'p99_time_taken_ms',
    'peak_hour_concentration',
    'protocol_consistency', 'range_time_interval_sec', 'response_size_entropy', 'robust_cv_client_bytes',
    'robust_cv_server_bytes',
    'robust_cv_time_interval_sec', 'robust_cv_time_taken_ms', 'safari_ratio', 'script_ratio', 'secure_transport_ratio',
    'skewness_time_interval_sec',
    'std_time_interval_sec', 'sum_time_interval_sec', 'suspicious_domain_ratio', 'suspicious_tld_ratio',
    'thirdparty_service_ratio',
    'tld_diversity', 'total_data_volume', 'xml_ratio'
]

app_str_non_numeric_feature_fields = ["domain"]

app_arr_non_numeric_feature_fields = [
    "http_methods",
    "http_statuses",
    "resp_content_types",
    "key_hostnames",
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
        # Make MultiHotEncoder available in __main__ module for pickle loading
        import __main__
        if not hasattr(__main__, 'MultiHotEncoder'):
            __main__.MultiHotEncoder = MultiHotEncoder
        
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
    
    # # Handle feature name compatibility issues
    # # Map new feature names to old ones for backward compatibility
    # feature_mapping = {
    #     'median_time_interval_sec': 'median_interval_sec',
    #     'std_time_interval_sec': 'std_interval_sec',
    #     'range_time_interval_sec': 'range_interval_sec',
    #     'avg_interval_sec': 'avg_interval_sec',  # This one might already be correct
    # }
    #
    # # Create mapped columns for backward compatibility
    # for new_name, old_name in feature_mapping.items():
    #     if new_name in features_og.columns and old_name not in features_og.columns:
    #         features_og[old_name] = features_og[new_name]
    
    # Filter to only include columns that actually exist in the data
    available_columns = [col for col in app_feature_fields if col in features_og.columns]
    features_og = features_og[available_columns]
    
    # Fill NaN values with appropriate defaults
    # For numeric columns, use 0
    numeric_columns = features_og.select_dtypes(include=[np.number]).columns
    features_og[numeric_columns] = features_og[numeric_columns].fillna(0)
    
    # For string columns, use empty string
    string_columns = features_og.select_dtypes(include=['object']).columns
    for col in string_columns:
        features_og[col] = features_og[col].fillna('')

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
                if hasattr(estimator, "named_steps") and "xgb" in estimator.named_steps:
                    classes = estimator.named_steps["xgb"].classes_
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
                and "feat_sel" in estimator.named_steps
            ):
                features_scaled = estimator["feat_sel"].transform(
                    estimator["ct"].transform(features_pd)
                )
            elif (
                hasattr(estimator, "named_steps") and "feat_sel" in estimator.named_steps
            ):
                features_scaled = estimator["feat_sel"].transform(
                    estimator["ct"].transform(features_pd)
                )
            else:
                # Try accessing as dictionary for backward compatibility
                if "feat_sel" in estimator:
                    features_scaled = estimator["feat_sel"].transform(
                        estimator["ct"].transform(features_pd)
                    )
                else:
                    features_scaled = estimator["feat_sel"].transform(
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
            
            # Generate text explanation using SHAP
            try:
                explainer = ModelExplainer(estimator, selected_feature_names, logger)
                
                # Convert observation to dict for the explainer
                observation_dict = observation_series.to_dict()
                
                # Generate text explanation
                text_explanation = explainer.generate_text_explanation(
                    features_scaled=features_scaled,
                    observation_index=observation_index,
                    observation_data=observation_dict,
                    predicted_class=predicted_class_name,
                    predicted_proba=predicted_class_proba,
                    top_n_features=constants.TOP_FEATURES_COUNT
                )
                
                # Save SHAP waterfall plot
                shap_plot_path = f"{parent_dir}shap_explanation.png"
                explainer.save_shap_plot(
                    features_scaled=features_scaled,
                    observation_index=observation_index,
                    predicted_class_index=predicted_class_index,
                    save_path=shap_plot_path,
                    max_display=20
                )
                
                # Save text explanation
                explanation_path = f"{parent_dir}explanation.txt"
                with open(explanation_path, 'w') as f:
                    f.write(text_explanation)
                    
                # Log if this is an anomaly
                if (predicted_class_name == 1) and (predicted_class_proba >= prob_cutoff):
                    logger.warning(f"🚨 ANOMALY DETECTED: {observation_series.get('domain', observation_key)} for {application} (probability: {predicted_class_proba:.3f})")
                    logger.warning(f"   Explanation: {text_explanation.split('.')[0]}")
                    
                logger.info(f"Generated explanation and saved to {parent_dir}")
                
            except Exception as e:
                logger.error(f"Failed to generate explanation: {e}")
                text_explanation = f"Unable to generate detailed explanation: {e}"

            full_predictions = sorted(
                [
                    {"class": str(c), "probability": str(round(float(100.0 * p), 4))}
                    for p, c in zip(predictions[observation_index], classes)
                ],
                key=lambda x: x["probability"],
                reverse=True,
            )
            full_predictions_path = f"{parent_dir}full_predictions.json"
            save_json_data(full_predictions, full_predictions_path)
            
            # Save explanation to JSON as well
            explanation_json = ExplanationJson(
                domain=observation_series.get("domain", observation_key),
                application=application,
                predicted_class=predicted_class_name,
                probability=predicted_class_proba,
                is_anomaly=bool(predicted_class_proba >= prob_cutoff),
                explanation=text_explanation,
                top_features=[]
            )
            
            # Try to get top features for JSON
            try:
                shap_values, _ = explainer.calculate_shap_values(
                    features_scaled, observation_index, predicted_class_index
                )
                top_features = explainer.get_top_features(shap_values, top_n=constants.TOP_FEATURES_COUNT)
                for feature_name, shap_value, feature_idx in top_features:
                    feature_value = features_scaled[observation_index, feature_idx]
                    explanation_json.top_features.append(
                        TopFeature(
                            feature=feature_name,
                            shap_value=shap_value,
                            feature_value=feature_value
                        )
                    )
            except:
                pass
                
            explanation_json_path = f"{parent_dir}explanation.json"
            save_json_data(explanation_json.model_dump(), explanation_json_path)


def detect_anomalous_domain_with_anomaly_model(
    input_path: str,
    custom_model_path: Path,
    app_prediction_dir: str,
    anomaly_threshold: float = -0.1,
) -> Dict[str, Any]:
    """
    Detect anomalous domains using a custom anomaly detection model.
    
    Args:
        input_path (str): The path to the input JSON file containing the data.
        custom_model_path (Path): The path to the custom anomaly model file.
        app_prediction_dir (str): The directory to save the predictions.
        anomaly_threshold (float): Threshold for anomaly score. Lower values = more anomalous.
        
    Returns:
        Dict[str, Any]: Detection results summary containing analyzed domains, anomalies found, etc.
    """
    logger = logging.getLogger(__name__)
    
    # Initialize detection results tracking
    detection_results = DetectionResults(
        model_used=str(custom_model_path.name),
        prob_cutoff_used=anomaly_threshold
    )
    
    # Load the anomaly detection model
    try:
        with open(custom_model_path, "rb") as _file:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=DeprecationWarning)
                raw_models = pickle.load(_file)
    except Exception as e:
        logger.error(f"Failed to load custom model {custom_model_path}: {e}")
        detection_results.success = False
        detection_results.error_message = f"Failed to load model: {e}"
        return detection_results.model_dump()

    # Convert single model to the expected format
    models = dict()
    apps = set()

    if isinstance(raw_models, list) and len(raw_models) > 0:
        raw_model = raw_models[0].copy()
        app = raw_model.pop("key")
        models[app] = raw_model
        apps.add(app)
    else:
        logger.error(f"Unexpected model format in {custom_model_path}")
        detection_results.success = False
        detection_results.error_message = "Unexpected model format"
        return detection_results.model_dump()

    # Load and convert features
    features_og, features_pd = convert_supply_chain_summaries_to_features(
        load_json_file(input_path)
    )

    for observation_index, observation_series in features_og.iterrows():
        application = observation_series["application"]
        domain = observation_series.get("domain", "unknown")
        
        # Track applications found
        if application not in detection_results.applications_found:
            detection_results.applications_found.append(application)
            
        if application not in models:
            logger.info(f"[x] Application not found in custom models: {application}")
            continue
        else:
            logger.info(f"[x] Testing {application} for supply chain anomalies")
            detection_results.total_domains_analyzed += 1
            observation_key = observation_series["key"]
            model = models[application]

            # Check if this is an anomaly detection model
            if model.get("model_type") != "ensemble_anomaly":
                logger.warning(f"Model for {application} is not an anomaly detection model. Skipping.")
                continue

            feature_transformer = model["feature_transformer"]
            estimator = model["ensemble_detector"]
            
            # Transform features for this observation
            observation_df = features_pd.iloc[[observation_index]]
            features_transformed = feature_transformer.transform(observation_df)
            
            # Get anomaly predictions and scores
            try:
                # Get prediction (-1 = anomaly, 1 = normal)
                anomaly_prediction = estimator.predict(features_transformed)[0]
                # Get anomaly score (lower = more anomalous)
                anomaly_score = estimator.decision_function(features_transformed)[0]
                
                is_anomaly = (anomaly_prediction == -1) or (anomaly_score < anomaly_threshold)
                
                # Create output directory
                obs_file_dir = (
                    str(observation_index)
                    + "_"
                    + observation_key.replace(" ", "_")
                    .replace("'", "")
                    .replace("/", "")[:35]
                )
                parent_dir = f"{app_prediction_dir}/{obs_file_dir}/"
                safe_create_path(parent_dir)
                
                # Generate explanation for anomaly detection
                if is_anomaly:
                    detection_results.anomalies_detected += 1
                    text_explanation = f"Anomaly detected for {domain}. Anomaly score: {anomaly_score:.4f} (threshold: {anomaly_threshold}). This indicates the application behavior deviates significantly from learned normal patterns, possibly indicating a supply chain compromise."
                    
                    anomaly_info = AnomalyInfo(
                        domain=domain,
                        application=application,
                        observation_key=observation_key,
                        predicted_class="anomaly",
                        probability=abs(anomaly_score),  # Use absolute score as proxy for confidence
                        prediction_index=observation_index,
                        explanation=text_explanation
                    )
                    detection_results.anomalous_domains.append(anomaly_info)
                    logger.warning(f"🚨 ANOMALY DETECTED: {domain} for {application} (score: {anomaly_score:.4f})")
                    logger.warning(f"   Explanation: {text_explanation.split('.')[0]}")
                else:
                    detection_results.normal_domains += 1
                    text_explanation = f"Normal behavior detected for {domain}. Anomaly score: {anomaly_score:.4f} (threshold: {anomaly_threshold}). The application behavior matches learned normal patterns."
                    logger.info(f"✅ Normal behavior: {domain} for {application} (score: {anomaly_score:.4f})")
                
                # Save explanation
                explanation_path = f"{parent_dir}explanation.txt"
                with open(explanation_path, 'w') as f:
                    f.write(text_explanation)
                
                # Save prediction details
                prediction_details = {
                    "domain": domain,
                    "application": application,
                    "anomaly_score": float(anomaly_score),
                    "anomaly_prediction": int(anomaly_prediction),
                    "is_anomaly": bool(is_anomaly),
                    "threshold_used": float(anomaly_threshold),
                    "explanation": text_explanation
                }
                
                prediction_path = f"{parent_dir}anomaly_prediction.json"
                save_json_data(prediction_details, prediction_path)
                
            except Exception as e:
                logger.error(f"Failed to process {domain} for {application}: {e}")
                continue

    detection_results.success = True
    logger.info(f"Anomaly detection completed. Analyzed {detection_results.total_domains_analyzed} domains, found {detection_results.anomalies_detected} anomalies")
    
    return detection_results.model_dump()


def detect_anomalous_domain_with_custom_model(
    input_path: str,
    custom_model_path: Path,
    app_prediction_dir: str,
    prob_cutoff: float = 0.8,
) -> Dict[str, Any]:
    """
    Detect anomalous domains using an individual custom model.

    Args:
        input_path (str): The path to the input JSON file containing the data.
        custom_model_path (Path): The path to the custom model file.
        app_prediction_dir (str): The directory to save the predictions.
        prob_cutoff (float): The cutoff for probability to determine if it's anomalous.

    Returns:
        Dict[str, Any]: Detection results summary containing analyzed domains, anomalies found, etc.
    """
    logger = logging.getLogger(__name__)
    
    # Initialize detection results tracking using Pydantic model
    detection_results = DetectionResults(
        model_used=str(custom_model_path.name),
        prob_cutoff_used=prob_cutoff
    )

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
        detection_results.success = False
        detection_results.error_message = f"Failed to load model: {e}"
        return detection_results.model_dump()

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
        detection_results.success = False
        detection_results.error_message = "Unexpected model format"
        return detection_results.model_dump()

    features_og, features_pd = convert_supply_chain_summaries_to_features(
        load_json_file(input_path)
    )

    for observation_index, observation_series in features_og.iterrows():
        application = observation_series["application"]
        domain = observation_series.get("domain", "unknown")
        
        # Track applications found
        if application not in detection_results.applications_found:
            detection_results.applications_found.append(application)
            
        if application not in models:
            logger.info(
                "[x] Application not found in custom models: " + str(application)
            )
        else:
            logger.info(
                "[x] Application found to test supply chain compromises against: "
                + str(application)
            )
            detection_results.total_domains_analyzed += 1
            observation_key = observation_series["key"]
            model = models[application]

            estimator = model["estimator"]
            selected_feature_names = model["selected_features"]
            
            # Get classes from the final classifier in the pipeline
            try:
                if hasattr(estimator, "named_steps") and "xgb" in estimator.named_steps:
                    classes = estimator.named_steps["xgb"].classes_
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
                and "feat_sel" in estimator.named_steps
            ):
                features_scaled = estimator["feat_sel"].transform(
                    estimator["ct"].transform(features_pd)
                )
            elif (
                hasattr(estimator, "named_steps") and "feat_sel" in estimator.named_steps
            ):
                features_scaled = estimator["feat_sel"].transform(
                    estimator["ct"].transform(features_pd)
                )
            else:
                # Try accessing as dictionary for backward compatibility
                if "feat_sel" in estimator:
                    features_scaled = estimator["feat_sel"].transform(
                        estimator["ct"].transform(features_pd)
                    )
                else:
                    features_scaled = estimator["feat_sel"].transform(
                        estimator["ct"].transform(features_pd)
                    )

            predicted_class_index = predictions[observation_index].argmax()
            predicted_class_name = classes[predicted_class_index]
            predicted_class_proba = predictions[
                observation_index, predicted_class_index
            ]
            
            # Create output directory
            obs_file_dir = (
                str(observation_index)
                + "_"
                + observation_key.replace(" ", "_")
                .replace("'", "")
                .replace("/", "")[:35]
            )
            parent_dir = f"{app_prediction_dir}/{obs_file_dir}/"
            safe_create_path(parent_dir)
            
            # Generate text explanation using SHAP
            try:
                explainer = ModelExplainer(estimator, selected_feature_names, logger)
                
                # Convert observation to dict for the explainer
                observation_dict = observation_series.to_dict()
                
                # Generate text explanation
                text_explanation = explainer.generate_text_explanation(
                    features_scaled=features_scaled,
                    observation_index=observation_index,
                    observation_data=observation_dict,
                    predicted_class=predicted_class_name,
                    predicted_proba=predicted_class_proba,
                    top_n_features=constants.TOP_FEATURES_COUNT
                )
                
                # Save SHAP waterfall plot
                shap_plot_path = f"{parent_dir}shap_explanation.png"
                explainer.save_shap_plot(
                    features_scaled=features_scaled,
                    observation_index=observation_index,
                    predicted_class_index=predicted_class_index,
                    save_path=shap_plot_path,
                    max_display=20
                )
                
                # Save text explanation
                explanation_path = f"{parent_dir}explanation.txt"
                with open(explanation_path, 'w') as f:
                    f.write(text_explanation)
                    
                logger.info(f"Generated explanation for {domain}: {text_explanation.split('.')[0]}")
                
            except Exception as e:
                logger.error(f"Failed to generate explanation: {e}")
                text_explanation = f"Unable to generate detailed explanation: {e}"
            
            # Check if this is an anomaly based on probability cutoff
            is_anomaly = bool(predicted_class_proba >= prob_cutoff)
            if is_anomaly:
                detection_results.anomalies_detected += 1
                anomaly_info = AnomalyInfo(
                    domain=domain,
                    application=application,
                    observation_key=observation_key,
                    predicted_class=predicted_class_name,
                    probability=predicted_class_proba,
                    prediction_index=observation_index,
                    explanation=text_explanation
                )
                detection_results.anomalous_domains.append(anomaly_info)
                logger.warning(f"🚨 ANOMALY DETECTED: {domain} for {application} (probability: {predicted_class_proba:.3f})")
                logger.warning(f"   Explanation: {text_explanation.split('.')[0]}")
            else:
                detection_results.normal_domains += 1
                logger.info(f"✅ Normal behavior: {domain} for {application} (probability: {predicted_class_proba:.3f})")

            full_predictions = sorted(
                [
                    PredictionClass(class_name=c, probability=100.0 * p).model_dump(by_alias=True)
                    for p, c in zip(predictions[observation_index], classes)
                ],
                key=lambda x: x["probability"],
                reverse=True,
            )
            full_predictions_path = f"{parent_dir}full_predictions.json"
            save_json_data(full_predictions, full_predictions_path)
            
            # Save explanation to JSON as well
            explanation_json = ExplanationJson(
                domain=domain,
                application=application,
                predicted_class=predicted_class_name,
                probability=predicted_class_proba,
                is_anomaly=is_anomaly,
                explanation=text_explanation,
                top_features=[]
            )
            
            # Try to get top features for JSON
            try:
                shap_values, _ = explainer.calculate_shap_values(
                    features_scaled, observation_index, predicted_class_index
                )
                top_features = explainer.get_top_features(shap_values, top_n=constants.TOP_FEATURES_COUNT)
                for feature_name, shap_value, feature_idx in top_features:
                    feature_value = features_scaled[observation_index, feature_idx]
                    explanation_json.top_features.append(
                        TopFeature(
                            feature=feature_name,
                            shap_value=shap_value,
                            feature_value=feature_value
                        )
                    )
            except:
                pass
                
            explanation_json_path = f"{parent_dir}explanation.json"
            save_json_data(explanation_json.model_dump(), explanation_json_path)

    # Return detection results summary
    logger.info(f"Detection completed: {detection_results.total_domains_analyzed} domains analyzed, "
                f"{detection_results.anomalies_detected} anomalies detected, "
                f"{detection_results.normal_domains} normal domains")
    
    return detection_results.model_dump()

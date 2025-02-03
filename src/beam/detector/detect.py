import logging
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
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
    "domain_cnt",
    "referer_domain_cnt",
    "key_hostname_cnt",
    "http_status_cnt",
    "http_method_cnt",
    "req_content_type_cnt",
    "resp_content_type_cnt",
    "cloud_traffic_pct",
    "web_traffic_pct",
    "refered_traffic_pct",
    "avg_time_interval_sec",
    "std_time_interval_sec",
    "median_time_interval_sec",
    "range_time_interval_sec",
    "range_timestamp",
    "key_hostnames",
    "http_methods",
    "http_statuses",
    "req_content_types",
    "resp_content_types",
]


def load_malware_model(combined_app_model_path):
    with open(combined_app_model_path, "rb") as _file:
        model = pickle.load(_file)[0]
    return model


def load_app_models(app_model_path):
    with open(app_model_path, "rb") as _file:
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

    def __init__(self):
        self.mlbs = list()
        self.n_columns = 0
        self.categories_ = self.classes_ = list()

    def fit(self, X: pd.DataFrame, y=None):
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

    def transform(self, X: pd.DataFrame):
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


def convert_summary_to_features(input_data):
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


def convert_supply_chain_summaries_to_features(input_data):
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
    input_path, combined_app_model_path, combined_app_prediction_directory
):
    """
    Detect anomalous applications in the given input data.

    Args:
        input_path (str): The path to the input JSON file containing the data.

    Returns:
        None

    Raises:
        None
    """
    logger = logging.getLogger(__name__)
    logger.info(f"[x] Detecting anomalous applications in {input_path}")

    model = load_malware_model(combined_app_model_path)
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
        parent_dir = f"{combined_app_prediction_directory}/{obs_file_dir}/"
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


def detect_anomalous_domain(input_path, app_model_path, app_prediction_dir):
    logger = logging.getLogger(__name__)
    apps, models = load_app_models(app_model_path)
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

            # plt.clf()
            # chosen_instance = features_scaled[observation_index, :]
            # explainer = shap.TreeExplainer(estimator['rf'])
            # shap_values = explainer.shap_values(chosen_instance)
            # shap.initjs()
            # exp = shap.Explanation(values=shap_values[..., predicted_class_index],
            #                        base_values=explainer.expected_value[predicted_class_index],
            #                        data=chosen_instance,
            #                        feature_names=selected_feature_names)

            # shap.waterfall_plot(exp, max_display=20, show=False)
            obs_file_dir = (
                str(observation_index)
                + "_"
                + observation_key.replace(" ", "_")
                .replace("'", "")
                .replace("/", "")[:35]
            )
            parent_dir = f"{app_prediction_dir}/{obs_file_dir}/"
            safe_create_path(parent_dir)
            # f"{parent_dir}{predicted_class_name}_shap_waterfall.png"
            exp_png_path = ""
            # plt.savefig(exp_png_path, dpi=150, bbox_inches='tight')

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

            if predicted_class_name == "negative_label":
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

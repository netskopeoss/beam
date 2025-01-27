import pickle
import shap
import logging

from beam import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer


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
            raise ValueError('Please fit the transformer first.')
        if self.n_columns != X.shape[1]:
            raise ValueError(f'The fit transformer deals with {self.n_columns} columns '
                             f'while the input has {X.shape[1]}.'
                            )
        result = list()
        for i in range(self.n_columns):
            result.append(self.mlbs[i].transform(X.iloc[:, i]))

        result = np.concatenate(result, axis=1)
        return result


def detect_anomalous_app(input_path: str, logger: logging.Logger):
    """
    Detect anomalous applications in the given input data.

    Args:
        input_path (str): The path to the input JSON file containing the data.
        logger (logging.Logger): Logger instance for capturing log messages.

    Returns:
        None

    Raises:
        None
    """
    logger.info(f"[x] Detecting anomalous applications in {input_path}")

    with open('../models/c2_model.pkl', 'rb') as _file:
        # TODO: Save this model off in probability mode
        classifier = pickle.load(_file)

    with open('../models/col_transform.pkl', 'rb') as _file:
        col_transformer = pickle.load(_file)

    with open('../models/feature_names.pkl', 'rb') as _file:
        feature_names = pickle.load(_file)

    meta_cols = [
        'key'
    ]
    feature_columns = meta_cols + [
        "max_time_taken_ms", "min_time_taken_ms", "sum_time_taken_ms", "avg_time_taken_ms",
        "std_time_taken_ms", "median_time_taken_ms", "range_time_taken_ms",
        "max_client_bytes", "min_client_bytes", "sum_client_bytes", "avg_client_bytes",
        "std_client_bytes", "median_client_bytes", "range_client_bytes",
        "max_server_bytes", "min_server_bytes", "sum_server_bytes", "avg_server_bytes",
        "std_server_bytes", "median_server_bytes", "range_server_bytes",
        "domain_cnt", "referer_domain_cnt", "key_hostname_cnt",
        "http_status_cnt", "http_method_cnt", "req_content_type_cnt", "resp_content_type_cnt",
        "cloud_traffic_pct", "web_traffic_pct", "refered_traffic_pct",
        "avg_time_interval_sec", "std_time_interval_sec", "median_time_interval_sec",
        "range_time_interval_sec", "range_timestamp",

        "key_hostnames",
        "http_methods",
        "http_statuses",
        "req_content_types",
        "resp_content_types",
    ]

    features_og = pd.json_normalize(utils.load_json_file(input_path))
    features_og.reset_index()
    features_og = features_og[feature_columns]
    feature_start_index = len(meta_cols)
    features = pd.DataFrame(data=features_og.to_numpy()[:, feature_start_index:],
                            columns=features_og.columns[feature_start_index:])
    features.reset_index()
    features_scaled = col_transformer.transform(features)
    # print("features_scaled", features_scaled)
    # print("features_scaled shape", features_scaled.shape)

    # Predicting the Test set results
    y_pred = classifier.predict_proba(features_scaled)

    for observation_index, observation_key in enumerate(features_og['key']):
        # Create Tree Explainer object that can calculate shap values
        predicted_class_index = y_pred[observation_index].argmax()
        predicted_class_name = classifier.classes_[predicted_class_index]
        predicted_class_proba = y_pred[observation_index, predicted_class_index]

        plt.clf()
        chosen_instance = features_scaled[observation_index, :]
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(chosen_instance)
        shap.initjs()
        exp = shap.Explanation(values=shap_values[..., predicted_class_index],
                               base_values=explainer.expected_value[predicted_class_index],
                               data=chosen_instance,
                               feature_names=feature_names)
        shap.waterfall_plot(exp,
                            max_display=20,
                            show=False)
        obs_file_dir = str(observation_index) + '_' + observation_key.replace(' ', '_').replace("'", "").replace("/", "")[:35]
        parent_dir = f"predictions/{obs_file_dir}/"
        utils.safe_create_path(parent_dir)
        exp_png_path = f"{parent_dir}{predicted_class_name}_shap_waterfall.png"
        plt.savefig(exp_png_path, dpi=150, bbox_inches='tight')

        full_predictions = sorted([
            {'class': c, 'probability': round(100.0 * p, 4)} for p, c in zip(y_pred[observation_index], classifier.classes_)],
            key=lambda x: x['probability'],
            reverse=True
        )
        full_predictions_path = f"{parent_dir}full_predictions.json"
        utils.save_json_data(full_predictions, full_predictions_path)

        logger.info(f"""
i = {observation_index}
Advertised user agent = {observation_key}
Predicted class = {predicted_class_name} ({round(predicted_class_proba * 100, 2)}%)
Top 3 predictions = {full_predictions[:3]}
SHAP explanation path = {exp_png_path}
Full predictions path = {full_predictions_path}
\n""")


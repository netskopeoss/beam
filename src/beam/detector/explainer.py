"""Model Explainer Module - Generates human-readable explanations for predictions"""

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

import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import shap
from numpy.typing import NDArray
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


# Feature groups for better explanation organization
FEATURE_GROUPS = {
    "traffic_volume": {
        "features": ["sum_client_bytes", "sum_server_bytes", "avg_client_bytes", 
                    "avg_server_bytes", "total_data_volume", "avg_bytes_per_request"],
        "description": "data transfer patterns"
    },
    "time_patterns": {
        "features": ["avg_time_interval_sec", "std_time_interval_sec", "median_time_interval_sec",
                    "burst_ratio", "interval_entropy", "interval_regularity", "night_activity_ratio"],
        "description": "timing and frequency patterns"
    },
    "domain_characteristics": {
        "features": ["domain_cnt", "domain_concentration", "domain_entropy", "domain_diversity",
                    "suspicious_domain_ratio", "new_domain_ratio", "external_domain_ratio"],
        "description": "domain communication patterns"
    },
    "http_behavior": {
        "features": ["http_status_cnt", "http_method_cnt", "error_ratio", "redirect_ratio",
                    "https_ratio", "http2_usage_ratio"],
        "description": "HTTP protocol behavior"
    },
    "content_types": {
        "features": ["req_content_type_cnt", "resp_content_type_cnt", "json_ratio", 
                    "html_ratio", "js_ratio", "executable_ratio", "script_ratio"],
        "description": "content type patterns"
    },
    "user_agent": {
        "features": ["ua_diversity", "ua_entropy", "bot_ratio", "suspicious_ua_ratio",
                    "ua_consistency", "automation_suspicion"],
        "description": "user agent characteristics"
    },
    "response_characteristics": {
        "features": ["max_time_taken_ms", "avg_time_taken_ms", "std_time_taken_ms",
                    "response_size_cv", "large_response_ratio"],
        "description": "server response patterns"
    }
}

# Feature interpretation rules
FEATURE_INTERPRETATIONS = {
    # Volume-based features
    "sum_server_bytes": {
        "high_positive": "unusually high data download volume",
        "high_negative": "significantly lower data download than typical",
        "unit": "bytes"
    },
    "sum_client_bytes": {
        "high_positive": "unusually high data upload volume", 
        "high_negative": "significantly lower data upload than typical",
        "unit": "bytes"
    },
    
    # Time-based features
    "burst_ratio": {
        "high_positive": "concentrated burst activity pattern",
        "high_negative": "evenly distributed activity",
        "threshold": 0.5
    },
    "night_activity_ratio": {
        "high_positive": "unusual nighttime activity",
        "high_negative": "normal daytime activity pattern",
        "threshold": 0.3
    },
    "interval_regularity": {
        "high_positive": "highly regular/automated timing pattern",
        "high_negative": "irregular human-like timing",
        "threshold": 0.8
    },
    
    # Domain features
    "suspicious_domain_ratio": {
        "high_positive": "high ratio of suspicious domains",
        "high_negative": "mostly trusted domains",
        "threshold": 0.1
    },
    "new_domain_ratio": {
        "high_positive": "high ratio of newly registered domains",
        "high_negative": "established domains only",
        "threshold": 0.2
    },
    "external_domain_ratio": {
        "high_positive": "high ratio of external domain communication",
        "high_negative": "mostly internal domain communication",
        "threshold": 0.5
    },
    
    # HTTP features
    "error_ratio": {
        "high_positive": "high error rate indicating issues",
        "high_negative": "low error rate indicating normal operation",
        "threshold": 0.1
    },
    "automation_suspicion": {
        "high_positive": "patterns suggesting automated/bot behavior",
        "high_negative": "normal human interaction patterns",
        "threshold": 0.5
    },
    
    # Content features
    "executable_ratio": {
        "high_positive": "high ratio of executable content",
        "high_negative": "minimal executable content",
        "threshold": 0.05
    },
    "script_ratio": {
        "high_positive": "high ratio of script content",
        "high_negative": "minimal script content", 
        "threshold": 0.1
    }
}


class ModelExplainer:
    """Generates human-readable explanations for model predictions using SHAP"""
    
    def __init__(self, model, feature_names: List[str], logger: Optional[logging.Logger] = None):
        """
        Initialize the ModelExplainer
        
        Args:
            model: The trained model (pipeline or estimator)
            feature_names: List of feature names
            logger: Optional logger instance
        """
        self.model = model
        self.feature_names = feature_names
        self.logger = logger or logging.getLogger(__name__)
        self.explainer = None
        
    def _get_base_estimator(self):
        """Extract the base estimator from a pipeline"""
        if hasattr(self.model, "named_steps"):
            # It's a pipeline - get the final classifier
            if "xgb_classifier" in self.model.named_steps:
                return self.model.named_steps["xgb_classifier"]
            elif "rf" in self.model.named_steps:
                return self.model.named_steps["rf"]
            else:
                # Get the last step
                return self.model.steps[-1][1]
        else:
            return self.model
            
    def _create_explainer(self, features_scaled: NDArray):
        """Create SHAP explainer if not already created"""
        if self.explainer is None:
            base_estimator = self._get_base_estimator()
            self.explainer = shap.TreeExplainer(base_estimator)
            
    def calculate_shap_values(self, features_scaled: NDArray, 
                            observation_index: int,
                            predicted_class_index: int) -> Tuple[NDArray, float]:
        """
        Calculate SHAP values for a specific observation
        
        Returns:
            Tuple of (shap_values, expected_value)
        """
        self._create_explainer(features_scaled)
        
        # Get SHAP values for the specific observation
        chosen_instance = features_scaled[observation_index, :].reshape(1, -1)
        shap_values = self.explainer.shap_values(chosen_instance)
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            shap_values_for_class = shap_values[predicted_class_index][0]
            expected_value = self.explainer.expected_value[predicted_class_index]
        else:
            shap_values_for_class = shap_values[0]
            expected_value = self.explainer.expected_value
            
        return shap_values_for_class, expected_value
        
    def get_top_features(self, shap_values: NDArray, top_n: int = 10) -> List[Tuple[str, float, float]]:
        """
        Get top features by absolute SHAP value
        
        Returns:
            List of tuples (feature_name, shap_value, feature_value)
        """
        # Get absolute values for ranking
        abs_shap_values = np.abs(shap_values)
        top_indices = np.argsort(abs_shap_values)[-top_n:][::-1]
        
        top_features = []
        for idx in top_indices:
            if idx < len(self.feature_names):
                feature_name = self.feature_names[idx]
                shap_value = shap_values[idx]
                top_features.append((feature_name, shap_value, idx))
                
        return top_features
        
    def _get_feature_group(self, feature_name: str) -> Optional[str]:
        """Get the feature group for a given feature"""
        for group_name, group_info in FEATURE_GROUPS.items():
            if feature_name in group_info["features"]:
                return group_name
        return None
        
    def _interpret_feature_impact(self, feature_name: str, shap_value: float, 
                                feature_value: float) -> str:
        """Generate human-readable interpretation for a feature's impact"""
        impact_direction = "increases" if shap_value > 0 else "decreases"
        
        if feature_name in FEATURE_INTERPRETATIONS:
            interp = FEATURE_INTERPRETATIONS[feature_name]
            
            # Determine if this is a high positive or negative contribution
            if shap_value > 0:
                interpretation = interp.get("high_positive", f"high {feature_name}")
            else:
                interpretation = interp.get("high_negative", f"low {feature_name}")
                
            # Add quantitative context if available
            if "threshold" in interp and not np.isnan(feature_value):
                if feature_value > interp["threshold"]:
                    context = f" ({feature_value:.2f}, above threshold {interp['threshold']})"
                else:
                    context = f" ({feature_value:.2f})"
            elif "unit" in interp and not np.isnan(feature_value):
                context = f" ({feature_value:,.0f} {interp['unit']})"
            else:
                context = ""
                
            return f"{interpretation}{context} {impact_direction} anomaly score"
        else:
            # Generic interpretation
            value_desc = "high" if feature_value > 0 else "low"
            return f"{value_desc} {feature_name} ({feature_value:.2f}) {impact_direction} anomaly score"
            
    def generate_text_explanation(self, 
                                features_scaled: NDArray,
                                observation_index: int,
                                observation_data: Dict[str, Any],
                                predicted_class: str,
                                predicted_proba: float,
                                top_n_features: int = 5) -> str:
        """
        Generate a human-readable text explanation for the prediction
        
        Args:
            features_scaled: Scaled feature matrix
            observation_index: Index of the observation
            observation_data: Original observation data (with metadata)
            predicted_class: The predicted class name
            predicted_proba: Prediction probability
            top_n_features: Number of top features to explain
            
        Returns:
            Human-readable explanation string
        """
        # Get application and domain info
        application = observation_data.get("application", "Unknown")
        domain = observation_data.get("domain", observation_data.get("key", "unknown domain"))
        
        # Calculate SHAP values
        # For binary classification, we explain the positive class (anomaly)
        predicted_class_index = 1 if predicted_class == "anomaly" else 0
        shap_values, expected_value = self.calculate_shap_values(
            features_scaled, observation_index, predicted_class_index
        )
        
        # Get top contributing features
        top_features = self.get_top_features(shap_values, top_n_features)
        
        # Start building the explanation
        if predicted_proba >= 0.8:
            severity = "high confidence"
        elif predicted_proba >= 0.6:
            severity = "moderate confidence"
        else:
            severity = "low confidence"
            
        explanation_parts = [
            f"Communication from {application} to {domain} is flagged for potential supply chain compromise with {severity} (probability: {predicted_proba:.1%})."
        ]
        
        # Group features by category
        feature_groups = {}
        for feature_name, shap_value, feature_idx in top_features:
            group = self._get_feature_group(feature_name)
            if group not in feature_groups:
                feature_groups[group] = []
            
            # Get the actual feature value
            feature_value = features_scaled[observation_index, feature_idx]
            feature_groups[group].append((feature_name, shap_value, feature_value))
        
        # Build grouped explanations
        key_factors = []
        for group, features in feature_groups.items():
            if group and group in FEATURE_GROUPS:
                group_desc = FEATURE_GROUPS[group]["description"]
                group_factors = []
                
                for feature_name, shap_value, feature_value in features:
                    interpretation = self._interpret_feature_impact(
                        feature_name, shap_value, feature_value
                    )
                    group_factors.append(interpretation)
                
                if group_factors:
                    key_factors.append(f"{group_desc}: {', '.join(group_factors)}")
        
        # Add key factors to explanation
        if key_factors:
            explanation_parts.append("\nKey indicators:")
            for i, factor in enumerate(key_factors[:3], 1):  # Limit to top 3 groups
                explanation_parts.append(f"{i}. {factor}")
        
        # Add contextual summary
        if predicted_proba >= 0.8:
            explanation_parts.append(
                f"\nThis pattern significantly deviates from typical {application} behavior "
                f"and warrants immediate investigation."
            )
        elif predicted_proba >= 0.6:
            explanation_parts.append(
                f"\nThis pattern shows some deviation from typical {application} behavior "
                f"and should be monitored."
            )
            
        return "\n".join(explanation_parts)
        
    def save_shap_plot(self, 
                      features_scaled: NDArray,
                      observation_index: int,
                      predicted_class_index: int,
                      save_path: str,
                      max_display: int = 20):
        """
        Generate and save SHAP waterfall plot
        
        Args:
            features_scaled: Scaled feature matrix
            observation_index: Index of the observation
            predicted_class_index: Index of predicted class
            save_path: Path to save the plot
            max_display: Maximum features to display
        """
        # Calculate SHAP values
        shap_values, expected_value = self.calculate_shap_values(
            features_scaled, observation_index, predicted_class_index
        )
        
        # Create SHAP explanation object
        chosen_instance = features_scaled[observation_index, :]
        exp = shap.Explanation(
            values=shap_values,
            base_values=expected_value,
            data=chosen_instance,
            feature_names=self.feature_names
        )
        
        # Create waterfall plot
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(exp, max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"SHAP plot saved to {save_path}")
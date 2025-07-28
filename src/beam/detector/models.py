"""Detector Models Module - Pydantic models for type safety"""

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

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


class PredictionClass(BaseModel):
    """Model for individual prediction class results"""
    model_config = ConfigDict(populate_by_name=True)  # Allow both 'class_name' and 'class' as input
    
    class_name: str = Field(alias="class")
    probability: str
    
    @field_validator('class_name', mode='before')
    @classmethod
    def convert_class_name(cls, v):
        """Convert numpy types to native Python types"""
        return str(v)
    
    @field_validator('probability', mode='before')
    @classmethod
    def convert_probability(cls, v):
        """Convert probability to string with proper formatting"""
        if isinstance(v, str):
            return v
        # Handle numpy float types
        return str(round(float(v), 4))


class TopFeature(BaseModel):
    """Model for top feature information"""
    feature: str
    shap_value: float
    feature_value: float
    
    @field_validator('shap_value', 'feature_value', mode='before')
    @classmethod
    def convert_to_float(cls, v):
        """Convert numpy types to native Python float"""
        return float(v)


class ExplanationJson(BaseModel):
    """Model for explanation JSON output"""
    domain: str
    application: str
    predicted_class: str
    probability: float
    is_anomaly: bool
    explanation: str
    top_features: List[TopFeature] = []
    
    @field_validator('predicted_class', mode='before')
    @classmethod
    def convert_predicted_class(cls, v):
        """Convert numpy types to native Python types"""
        return str(v)
    
    @field_validator('probability', mode='before')
    @classmethod
    def convert_probability(cls, v):
        """Convert numpy types to native Python float"""
        return float(v)
    
    @field_validator('is_anomaly', mode='before')
    @classmethod
    def convert_is_anomaly(cls, v):
        """Convert numpy bool_ to native Python bool"""
        return bool(v)


class AnomalyInfo(BaseModel):
    """Model for anomaly information"""
    domain: str
    application: str
    observation_key: str
    predicted_class: str
    probability: float
    prediction_index: int
    explanation: str
    
    @field_validator('predicted_class', mode='before')
    @classmethod
    def convert_predicted_class(cls, v):
        """Convert numpy types to native Python types"""
        return str(v)
    
    @field_validator('probability', mode='before')
    @classmethod
    def convert_probability(cls, v):
        """Convert numpy types to native Python float"""
        return float(v)
    
    @field_validator('prediction_index', mode='before')
    @classmethod
    def convert_prediction_index(cls, v):
        """Convert numpy int64 to native Python int"""
        return int(v)


class DetectionResults(BaseModel):
    """Model for detection results summary"""
    model_used: str
    total_domains_analyzed: int = 0
    anomalies_detected: int = 0
    normal_domains: int = 0
    applications_found: List[str] = []
    anomalous_domains: List[AnomalyInfo] = []
    prob_cutoff_used: float
    success: bool = True
    error_message: Optional[str] = None
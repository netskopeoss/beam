# Custom App Models for BEAM

This directory stores custom app models created by users. BEAM can now learn and build models for custom apps beyond the 8 pre-trained ones.

## Complete Guide: Adding Custom App Models

### 📊 Data Requirements

#### Minimum Data Requirements:

1. **Transaction Threshold**: 
   - **Minimum**: 50 transactions per user agent (configurable in `ModelTrainer`)
   - **Recommended**: 100+ transactions (set in `constants.MIN_APP_TRANSACTIONS`)
   - **Optimal**: 500+ transactions for robust model training

2. **Data Quality Requirements**:
   - Must contain **user agent strings** for the target application
   - Traffic should span multiple sessions/time periods
   - Include diverse network behaviors (different domains, HTTP methods, etc.)
   - Representative of normal application usage patterns

3. **File Format Support**:
   - **PCAP files** (.pcap) - Network packet captures
   - **HAR files** (.har) - HTTP Archive files from browser dev tools
   - **Enriched JSON** - Pre-processed BEAM event files

### 🎯 Step-by-Step Instructions

#### Method 1: Training from PCAP/HAR Files (Recommended)

This is the most common method for training custom models from network traffic data.

##### Step 1: Prepare Your Training Data
```bash
# Create input directory structure (if not exists)
mkdir -p data/input

# Copy your PCAP or HAR files to the input directory
cp /path/to/your/app_traffic.pcap data/input/
# OR
cp /path/to/your/app_traffic.har data/input/
```

##### Step 2: Train the Model
BEAM automatically discovers applications in your traffic data and uses Docker with TensorFlow when available:
```bash
# Basic training command (auto-discovers all apps)
uv run python -m beam --train -i data/input/

# Train from a specific file
uv run python -m beam --train -i data/input/app_traffic.pcap

# Note: Training automatically uses Docker container with TensorFlow support if Docker is available.
# Falls back to local training (without Autoencoder) if Docker is not available.
```

##### Step 4: Verify Model Creation
```bash
# Check if model was created
ls -la models/custom_models/

# Check for combined model (includes pre-trained + custom)
ls -la models/combined_app_model.pkl
```

##### Step 5: Test the Model
```bash
# Run detection with custom models enabled (default)
python -m beam -i data/input/test_traffic.pcap --use_custom_models

# Check predictions directory for results
ls -la predictions/anomalous_app/
```

#### Method 2: Training from Pre-enriched Events

If you already have enriched event data:

```python
from beam.detector.trainer import train_custom_app_model, extract_app_features

# Extract features from enriched events
features_path = extract_app_features(
    input_data_path="data/enriched_events/your_data.json",
    output_path="data/app_summaries/your_app_features.json",
    min_transactions=100,  # Adjust based on your data
    fields=["useragent"]
)

# Train custom model (app_name will be discovered from features data)
train_custom_app_model(
    features_path=features_path,
    app_name="YourAppName",  # This is read from the features data
    output_model_path="models/custom_models/your_app_model.pkl",
    n_features=150,
    min_transactions=50
)
```

#### Method 3: Programmatic Training

For advanced users who want full control:

```python
from beam.detector.trainer import ModelTrainer
from beam.detector.utils import load_json_file

# Load your enriched events data
events_data = load_json_file("data/enriched_events/your_data.json")

# Create trainer with custom parameters
trainer = ModelTrainer(
    n_features=150,      # Number of features to select
    min_transactions=50  # Minimum transactions required
)

# Extract features for training
features_df = trainer.extract_features_for_training(events_data, "YourAppName")

# Convert to training format
training_data = features_df.to_dict('records')

# Train the model
model_info = trainer.train_model(training_data, "YourAppName")

# Save the model
if model_info:
    trainer.save_model(model_info, "models/custom_models/your_app_model.pkl")
```

### ⚙️ Configuration Options

#### Model Training Parameters:

```python
# Available parameters for ModelTrainer
trainer = ModelTrainer(
    n_features=150,      # Number of features to select (default: 150)
    min_transactions=50  # Minimum transactions per app (default: 50)
)

# Available parameters for training functions
train_custom_app_model(
    features_path="path/to/features.json",
    app_name="YourApp",
    output_model_path="path/to/model.pkl",
    n_features=150,         # Features to select
    min_transactions=50     # Minimum transactions
)
```

#### Feature Extraction Parameters:

```python
extract_app_features(
    input_data_path="data/enriched_events/data.json",
    output_path="data/app_summaries/features.json", 
    min_transactions=100,    # Minimum transactions (default: 50)
    fields=["useragent"]     # Fields to use for aggregation
)
```

### 🔍 Data Quality Validation

Check your data quality before training:

```python
# Quick validation script
import json

def validate_training_data(enriched_events_file):
    """Validate your training data quality."""
    
    with open(enriched_events_file, 'r') as f:
        events = json.load(f)
    
    print(f"Total events: {len(events)}")
    
    # Check user agents
    user_agents = [e.get('useragent', '') for e in events if e.get('useragent')]
    unique_uas = set(user_agents)
    
    print(f"Events with user agents: {len(user_agents)}")
    print(f"Unique user agents: {len(unique_uas)}")
    
    # Check for target app user agent
    target_patterns = ["YourApp", "your-app", "customapp"]  # Adjust patterns
    target_events = [e for e in events if any(pattern.lower() in e.get('useragent', '').lower() for pattern in target_patterns)]
    
    print(f"Events likely from target app: {len(target_events)}")
    
    if len(target_events) < 100:
        print("⚠️  Warning: Less than 100 target app events found")
        print("Consider collecting more data for better model performance")
    
    return len(target_events) >= 50

# Usage
validate_training_data("data/enriched_events/your_data.json")
```

### 🎯 Best Practices

#### 1. Data Collection Guidelines:
- **Diverse Traffic**: Collect traffic across different user sessions
- **Time Span**: Gather data over multiple days/weeks if possible  
- **Normal Usage**: Ensure data represents typical application behavior
- **Clean Data**: Remove obvious anomalies or testing artifacts

#### 2. App Naming Conventions:
- Use descriptive, unique names: `"Slack_Desktop"`, `"Custom_CRM_Tool"`
- Avoid spaces and special characters: Use underscores or hyphens
- Be consistent with naming across your organization

#### 3. Model Validation:
```bash
# Test your model with known good data
python -m beam.run -i data/input/validation_traffic.pcap --use_custom_models

# Check prediction accuracy in predictions/anomalous_app/
```

#### 4. Storage Management:
```bash
# Models are stored in:
models/custom_models/           # Individual custom models
models/combined_app_model.pkl   # Combined pre-trained + custom

# Keep backups of important models
cp models/combined_app_model.pkl models/backups/combined_$(date +%Y%m%d).pkl
```

### 🚨 Troubleshooting

#### Common Issues:

1. **"No training data with sufficient transactions"**
   - **Solution**: Reduce `min_transactions` parameter or collect more data
   - **Check**: Verify your data contains the target application's user agents

2. **"No features extracted"**
   - **Solution**: Check that enriched events file contains required fields
   - **Verify**: User agent strings are present and not empty

3. **Model training fails**
   - **Check**: Ensure XGBoost is installed: `pip install xgboost==2.1.1`
   - **Verify**: Input data format matches expected schema

4. **Poor detection performance**
   - **Solution**: Collect more diverse training data (500+ transactions recommended)
   - **Check**: Ensure training data is representative of normal app behavior

### 📈 Model Architecture

BEAM uses an ensemble anomaly detection approach for supply chain compromise detection:

```
Raw Data → Feature Extraction → ColumnTransformer → Ensemble Anomaly Detection → Trained Model
```

- **Isolation Forest**: Detects anomalies by isolating outliers in tree structures
- **One-Class SVM**: Learns boundaries around normal application behavior
- **Autoencoder**: Neural network-based reconstruction error detection (optional)
- **Ensemble Voting**: Combines multiple methods for robust anomaly detection
- **150+ Features**: Comprehensive network behavior analysis including timing, content types, domains, and traffic patterns

**Key Advantage**: Instead of learning "what is this app vs others", the model learns "what is normal behavior for this app" and detects deviations that could indicate supply chain compromises.

### 📊 Model Outputs

The training process produces the following files:

1. **Individual anomaly detection models** - Saved in this directory with a name based on the app name
   - Each model learns normal behavior patterns for a specific application
   - Models detect deviations from normal patterns (potential supply chain attacks)
   - Output: Anomaly scores and predictions instead of binary classifications

2. **Model Type**: `ensemble_anomaly` - Uses Isolation Forest, One-Class SVM, and optional Autoencoder

When custom models are available, BEAM automatically detects model type and uses appropriate detection logic.

### 🚀 Using Custom Models

By default, BEAM will use custom models alongside the pre-trained ones during detection. You can explicitly enable or disable this with the `--use_custom_models` flag:

```bash
# Use custom models (default)
python -m beam.run -i /path/to/pcap_directory --use_custom_models

# Ignore custom models
python -m beam.run -i /path/to/pcap_directory --no-use_custom_models
```

### 📊 Next Steps

After creating your custom model:

1. **Test thoroughly** with known traffic samples
2. **Monitor performance** in production
3. **Retrain periodically** as the application evolves
4. **Document** your model parameters and training data sources
5. **Share models** with your team using the combined model approach

Your custom model will now be automatically used alongside BEAM's pre-trained models for enhanced application detection!

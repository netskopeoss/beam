![BEAM image](https://github.com/netskopeoss/beam/blob/911595b4fd969d6305c0ba223084b7e6ae9568de/beam.jpg)

# Netskope BEAM
Behavioral Evaluation of Application Metrics (BEAM) is a Python library for detecting supply chain compromises by analyzing network traffic.

## Usage
### Prequisites
1. Install Zeek (formerly known as Bro) locally, using the instructions available [here](https://docs.zeek.org/en/current/install.html).

2. Clone the BEAM repo:
```bash
git clone git@github.com:netskopeoss/beam.git
```
### Install and run BEAM
1. Install via pip from the directory where you cloned the repo:

```bash
pip install -e .
```

2. Navigate to `beam/src` and run:

```bash
# Run BEAM in standard detection mode
python -m beam.run

# Run BEAM with custom models (default behavior)
python -m beam.run --use_custom_models

# Run BEAM with only the pre-trained models
python -m beam.run --no-use_custom_models
```

### Training Custom App Models

BEAM comes with 8 pre-trained application models, but you can train your own custom models for additional applications:

```bash
# Train a model for a new custom app using input files in the default directory
python -m beam.run --train --app_name "MyCustomApp"

# Specify a custom input directory
python -m beam.run --train --app_name "MyCustomApp" -i /path/to/pcap_directory

# Specify a custom output path for the model
python -m beam.run --train --app_name "MyCustomApp" --model_output /path/to/my_model.pkl
```

Once trained, custom models are automatically used alongside the pre-trained models during detection.

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

##### Step 2: Set App Name
Choose a descriptive name for your application:
```bash
export APP_NAME="MyCustomApp"  # Replace with your app name
```

##### Step 3: Train the Model
```bash
# Basic training command
python -m beam.run --train --app_name "$APP_NAME" -i data/input/

# With custom output path
python -m beam.run --train \
    --app_name "$APP_NAME" \
    -i data/input/ \
    --model_output "models/custom_models/${APP_NAME}_model.pkl"
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
python -m beam.run -i data/input/test_traffic.pcap --use_custom_models

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

# Train custom model
train_custom_app_model(
    features_path=features_path,
    app_name="YourAppName",
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

BEAM uses a hybrid machine learning approach for optimal performance:

```
Raw Data → Feature Extraction → ColumnTransformer → RF Feature Selection → XGBoost (Binary) → Trained Model
```

- **RandomForest Feature Selection**: Robust feature importance ranking and selection
- **XGBoost Classification**: Advanced gradient boosting for superior accuracy
- **Binary Classification**: Each model distinguishes one application from all others
- **150+ Features**: Comprehensive network behavior analysis including timing, content types, domains, and traffic patterns

### 📊 Next Steps

After creating your custom model:

1. **Test thoroughly** with known traffic samples
2. **Monitor performance** in production
3. **Retrain periodically** as the application evolves
4. **Document** your model parameters and training data sources
5. **Share models** with your team using the combined model approach

Your custom model will now be automatically used alongside BEAM's pre-trained models for enhanced application detection!

## Output from BEAM
BEAM generates multiple files and provides the following output:

1. The conclusion made from the provided PCAP or HAR files will be shown in the console with an associated probability of compromise.

2. For additional information, check the directories for each session under `beam/predictions` for [SHAP Waterfall plots](https://shap.readthedocs.io/en/latest/generated/shap.plots.waterfall.html). Each session has its own SHAP Waterfall image file, which show what features were used to determine BEAM's conclusion about the session.

**We included one sample HAR file in this repo so you can try BEAM immediately without adding any of your own data.**

### Sample Console Output
 Below is an example of the output generated from our sample HAR file:
![Console screenshot showing BEAM's output](https://github.com/netskopeoss/beam/blob/7040781dddfc1aca5d7c1d6dfcc132139cace731/beam_sample_console_screenshot.jpg)

In the screenshot above, you can see that the HAR file primarily contained traffic from Chrome and Box. The traffic from Box was compared against BEAM’s models. For the first two observations, the traffic was as expected, however for the last session, BEAM flagged it as “Potential supply chain compromise found”.

BEAM determined that there was a 95% possibility of a compromise here because the traffic in the HAR file showed communication from this Box application to an unusual endpoint (xqpt5z.dagmawi.io). It did this by flagging patterns in the traffic that did not match how a typical Box application communicates.

### Sample SHAP Waterfall plot

![SHAP Waterfall plot showing features for Box compromise](https://github.com/netskopeoss/beam/blob/97bdd3bce1b3f613fc07808608298a9529eb32f4/sample_shap_waterfall.jpg)

BEAM provides a SHAP Waterfall plot for each session analyzed, as shown above. The plot shows the reasoning behind the prediction via an impact breakdown of the evidence provided by each feature on the model’s output. In this particular case, the plot above shows the following reasons that this session was indicative of a compromise:
- not using the content type ‘application/json’
- not reaching out to Box’s servers
- the time taken for the requests
- the amount of data being received


## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[BSD 3-Clause](https://choosealicense.com/licenses/bsd-3-clause/)

# Custom App Models for BEAM

This directory stores custom app models created by users. BEAM can now learn and build models for custom apps beyond the 8 pre-trained ones.

## Training a Custom App Model

To train a custom app model, use the `--train` flag along with the `--app_name` parameter to specify the application name:

```
python -m beam.run --train --app_name "MyCustomApp" -i /path/to/pcap_directory
```

This will:
1. Process the input file (PCAP or HAR)
2. Extract features for the app
3. Train a model using the same feature set as the pre-trained models
4. Save the model to this directory
5. Create a combined model that includes both pre-trained and custom models

## Using Custom Models

By default, BEAM will use custom models alongside the pre-trained ones during detection. You can explicitly enable or disable this with the `--use_custom_models` flag:

```
# Use custom models (default)
python -m beam.run -i /path/to/pcap_directory --use_custom_models

# Ignore custom models
python -m beam.run -i /path/to/pcap_directory
```

## Model Outputs

The training process produces the following files:

1. **Individual app model** - Saved in this directory with a name based on the app name
2. **Combined app model** - Saved in the parent models directory as `combined_app_model.pkl`

When custom models are available, BEAM automatically uses the combined model for detection.

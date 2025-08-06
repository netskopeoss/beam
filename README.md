![BEAM image](https://github.com/netskopeoss/beam/blob/911595b4fd969d6305c0ba223084b7e6ae9568de/beam.jpg)

# Netskope BEAM
Behavioral Evaluation of Application Metrics (BEAM) is a Python library for detecting supply chain compromises by analyzing network traffic.

## 🚀 Quick Start (Recommended)

**The fastest way to see BEAM in action:**

```bash
# Clone the repository
git clone git@github.com:netskopeoss/beam.git
cd beam

# Install with uv (recommended)
uv sync
uv pip install -e .

# Run the interactive demo (one command!)
uv run python -m beam demo
```

This will:
- Automatically start required Docker services in the background
- Run the supply chain compromise detection demo
- Show you how BEAM detects malicious behavior in network traffic
- Complete in ~30 seconds

**What you'll see:** A real-world example of the Box cloud storage app infected with malware, and how BEAM's AI detects the hidden malicious communication.

**Results from the Demo:** Check the `data/demo_temp` directory for the artifacts containing the results.

## 🔧 Installation & Setup

### Prerequisites
1. **Python 3.12+** and **uv** (recommended) or pip
2. **An app for running containers** - Docker Desktop or an alternative for running Zeek, TensorFlow, and Ollama

## 🛡️ Running BEAM

BEAM uses a hybrid architecture: Python runs natively for performance while Docker handles infrastructure services automatically.

### Run Detection on Your Data

```bash

# Run detection on a specific file and only use the pre-packaged models
uv run python -m beam -i /path/to/traffic.har

# Run detection on a specific file with custom trained models
uv run python -m beam --use_custom_models -i /path/to/traffic.har
```

### Train Custom Models

BEAM automatically discovers applications in your traffic and trains models for any with sufficient data:

```bash

# Train from a specific file (auto-discovers all apps)
uv run python -m beam --train -i /path/to/traffic.har

```

## ⚙️ Configuration

### Environment Variables

By default, BEAM will use the local Llama container for mapping.

```bash
# Use Google Gemini for mapping
export GEMINI_API_KEY="your_api_key_here"

```

**For detailed instructions, data requirements, troubleshooting, and advanced configuration options, see the complete guide in [`models/custom_models/README.md`](models/custom_models/README.md).**

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

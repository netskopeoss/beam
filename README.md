![BEAM image](https://github.com/netskopeoss/beam/blob/911595b4fd969d6305c0ba223084b7e6ae9568de/beam.jpg)

# Netskope BEAM
Behavioral Evaluation of Application Metrics (BEAM) is a Python library for detecting supply chain compromises by analyzing network traffic.

## 🚀 Quick Start with Docker (Recommended)

**The fastest way to see BEAM in action:**

```bash
# Clone the repository
git clone git@github.com:netskopeoss/beam.git
cd beam

# Run the interactive demo (one command!)
./demo.sh
```

This will:
- Automatically build the Docker container with all dependencies
- Run the supply chain compromise detection demo
- Show you how BEAM detects malicious behavior in network traffic
- Complete in ~30 seconds

**What you'll see:** A real-world example of the Box cloud storage app infected with malware, and how BEAM's AI detects the hidden malicious communication.

## 🐳 Using BEAM with Docker

**Running BEAM is as easy as the demo!** Just use `./beam.sh`:

### Training Custom Models

BEAM automatically discovers applications in your traffic and trains models for any with sufficient data:

```bash
# Train models from a specific file (auto-discovers all apps)
./beam.sh --train --input /path/to/traffic.har

# Train models from a directory of PCAP/HAR files
./beam.sh --train --input /path/to/traffic_files/

# Train using files already in data/input/
./beam.sh --train
```

### Running Detection

```bash
# Run detection on files in data/input/
./beam.sh

# Run detection on a specific file
./beam.sh --input /path/to/suspicious_traffic.har

# Explicitly use custom models
./beam.sh --use_custom_models
```

BEAM automatically uses custom models if available, otherwise falls back to pre-trained models.

The `beam.sh` script handles all Docker complexity for you - no need to know Docker commands!

For advanced Docker usage, see [Docker Setup Guide](docker/README.md).

## Manual Installation

### Prerequisites
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
python -m beam
```

BEAM automatically uses custom models if available, otherwise uses pre-trained models.

### Training Custom App Models

BEAM comes with 8 pre-trained application models, but you can train your own custom models for additional applications. BEAM automatically discovers applications in your traffic data:

```bash
# Train models for all apps found in the default input directory
python -m beam --train

# Specify a custom input directory
python -m beam --train -i /path/to/pcap_directory
```

Once trained, custom models are automatically used alongside the pre-trained models during detection.

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

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
python beam
```
## Output from BEAM
1. The conclusion made from the provided PCAP or HAR files will be shown in the console with an associated probability of compromise.

2. For additional information, check the directories for each session under `beam/predictions` for [SHAP Waterfall plots](https://shap.readthedocs.io/en/latest/generated/shap.plots.waterfall.html). Each session has its own SHAP Waterfall image file, which show what features were used to determine BEAM's conclusion about the session.

The example below was produced from the sample HAR file included in this repo for a compromised version of Box:
![SHAP Waterfall plot showing features for Box compromise](https://github.com/netskopeoss/beam/blob/97bdd3bce1b3f613fc07808608298a9529eb32f4/sample_shap_waterfall.jpg)

The plot shows the reasoning behind the prediction via an impact breakdown of the evidence provided by each feature on the model’s output. In this particular case, the plot above shows the following reasons that this session was indicative of a compromise:
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

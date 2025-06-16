# BEAM Demo - Supply Chain Compromise Detection

This demo showcases BEAM's advanced security analysis capabilities using real network traffic data that contains a supply chain compromise.

## Demo Data

- **beacon_02_17_2025_18_12_39.har**: Network traffic capture from Box application usage containing a supply chain compromise

## What the Demo Shows

The demo will analyze the network traffic and demonstrate:

1. **Protocol-Level Security Analysis**: TLS/HTTPS usage, certificate patterns
2. **HTTP Header Fingerprinting**: User-Agent analysis, bot detection  
3. **Supply Chain Indicators**: Dependency tracking, suspicious domains
4. **Behavioral Baselines**: Error patterns, data volume analysis
5. **Security Insights**: Automated detection of supply chain compromises

## Expected Results

The analysis will detect:
- **Supply Chain Compromise**: Domain `xqpt5z.dagmawi.io` with high automation suspicion
- **Cross-Domain Activity**: Multiple cross-origin request patterns
- **Security Assessment**: Overall HIGH risk level

## Running the Demo

```bash
python -m beam demo
```

This will process the demo data and generate a comprehensive security analysis report.
# ML-WAF: Evolutionary Algorithm for WAF Bypass

An ML-driven approach to testing Web Application Firewalls against SQL injection attacks. Based on the academic paper "ML-Driven: An effective evolutionary algorithm to test Web Application Firewalls", this project uses a (+λ) evolutionary algorithm, a context-free grammar, and custom-built decision trees to automatically discover evasive attack payloads.

## Architecture

The system generates payloads via a context-free grammar, tests them against a real WAF target, and uses a RandomTree/RandomForest classifier to learn which patterns bypass the rules. These learned patterns (path conditions) then guide the adaptive mutation engine to generate increasingly effective attacks.

- **Grammar & Sampler**: Defines the SQLi search space and generates random initial payloads.
- **WAF Connector**: Interfaces with the real target (DVWA behind ModSecurity).
- **Slice Extractor**: Converts attack derivation trees into binary feature matrices.
- **Classifier**: Pure-Python implementation of RandomTree and RandomForest (no external ML libraries used).
- **EA Loop**: The main evolutionary algorithm (Variants B, D, and E).

## Dependencies

The project is written in pure Python 3.9+ to closely mirror the academic paper without relying on opaque ML libraries.

### Python Requirements
- `python >= 3.9`
- `matplotlib` (Only required for running `benchmark.py` to generate the comparison graphs)

Install Python dependencies:
```bash
pip install matplotlib
```

### Infrastructure Requirements
- Docker and Docker Compose (to run the real DVWA and ModSecurity WAF targets)

## Setup Guide

### 1. Start the WAF Environment

The target is a deliberately vulnerable web app (DVWA) protected by ModSecurity running the OWASP Core Rule Set.

```bash
cd ModSec_demo
docker compose up -d
```

This starts 3 endpoints:
- `http://localhost:9001`: Raw DVWA (No WAF)
- `http://localhost:9002`: ModSecurity (No Rules)
- `http://localhost:9003`: ModSecurity + OWASP CRS (**The Target**)

### 2. Configure DVWA

1. Open your browser and navigate to `http://localhost:9003`
2. Login with credentials: `admin` / `password`
3. Click "Create / Reset Database"
4. Go to **DVWA Security** in the left menu.
5. Set the security level to **Low** and click Submit.
6. Open your browser's Developer Tools (F12) -> Application/Storage tab.
7. Find the `PHPSESSID` cookie and copy its value.

## Usage

### Testing the WAF Interactively

Before running the full ML pipeline, verify your session is working:

```bash
cd WAF_model
# Open test_waf.py and update the `phpsessid` variable with your copied cookie, then run:
python test_waf.py
```

### Running the Evolutionary Algorithm

Run the main (µ+λ) EA loop (Variant E is enabled by default):

```bash
cd WAF_model
# You can pass the session ID directly via command line
python ea_loop.py <YOUR_PHPSESSID>
```

Useful arguments for `ea_loop.py`:
- `mock`: Run completely offline against a naive keyword blacklist instead of the real Docker WAF.
  - Example: `python ea_loop.py mock E forest`

### Running the Benchmark

Compare the ML-Driven variant E against the purely Random (RAN) baseline to see the learning advantage:

```bash
python benchmark.py --phpsessid <YOUR_PHPSESSID> --time 15 --output results.png
```

This will run both strategies for a 15-minute HTTP budget and save a cumulative bypass graph to `results.png`.

## Core Project Modules (`WAF_model/`)

- `ea_loop.py`: Main Evolutionary Algorithm runner
- `classifier.py`: Custom ML models (RandomTree, RandomForest)
- `grammar_definition.py`: SQL injection attack schemas and obfuscations
- `mutation.py`: Adaptive mutant generation
- `random_sampler.py`: Core payload generator and RAN baseline
- `slice_extractor.py`: Converts trees to ML feature matrices
- `waf_connector.py`: The HTTP wrapper around DVWA/ModSec
- `benchmark.py`: Comparative evaluator and plotting script

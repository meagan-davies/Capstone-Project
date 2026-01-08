# EMG+IMU Prosthetic Hand Control System

Real-time gesture classification for prosthetic hand control using EMG and IMU sensors with Delsys Trigno system.

## Features

- **Multi-class gesture recognition**: Neutral, Pinching, Grasping, Zipping
- **Real-time classification**: <100ms latency
- **Comprehensive feature extraction**: 8 EMG + 25 IMU features per sensor
- **Official Delsys API integration**: Uses DelsysAPI with AeroPy layer

## Installation

### 1. Clone repository
```bash
git clone 
cd emg-prosthetic-control
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
pip install -e .  # Install as editable package
```

### 4. Setup Delsys API
1. Place DelsysAPI files in `resources/DelsysAPI/`
2. Add your API key to `resources/delsys_key.txt`
3. Add your license to `resources/delsys_license.lic`

See [Installation Guide](docs/installation.md) for detailed instructions.

## Quick Start

### Training a Model
```bash
python scripts/train_model.py --data-dir data/raw/20251202 --model-name my_model
```

### Real-time Classification
```bash
python scripts/realtime_classify.py --model-name my_model
```

## Project Structure

- `src/`: Core library code
  - `data/`: Data loading and preprocessing
  - `features/`: Feature extraction
  - `models/`: Model training and evaluation
  - `realtime/`: Real-time classification system
- `scripts/`: Command-line tools
- `notebooks/`: Jupyter notebooks for analysis
- `config/`: Configuration files
- `docs/`: Documentation

## Data Collection Protocol

See [Data Collection Protocol](docs/data_collection_protocol.md) for details on:
- Sensor placement
- Trial structure
- File naming convention

## Documentation

- [Installation Guide](docs/installation.md)
- [API Reference](docs/api_reference.md)
- [Troubleshooting](docs/troubleshooting.md)
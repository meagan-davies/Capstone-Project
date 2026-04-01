# Usage Guide

## Table of Contents
- [Training a Model](#training-a-model)
- [Real-Time Classification](#real-time-classification)
- [Motor Control](#motor-control)
- [Sensor Management](#sensor-management)
- [Data Organization](#data-organization)

---

## Training a Model

### Quick Start - Single Session

Train a model with default settings on one data folder:
```bash
python prosthetic_control/scripts/train_model.py --data data/raw/prosthetic-control/20251202
```

### Training on Multiple Sessions

Combine data from multiple collection sessions:
```bash
python prosthetic_control/scripts/train_model.py \
    --data \
        data/raw/prosthetic-control/20251113 \
        data/raw/prosthetic-control/20251201 \
        data/raw/prosthetic-control/20251202 \
        data/raw/prosthetic-control/20260115 \
        data/raw/prosthetic-control/20260115_1 \
        data/raw/prosthetic-control/20260115_2 \
        data/raw/prosthetic-control/20260122 \
        data/raw/prosthetic-control/20260303 \
        data/raw/prosthetic-control/20260310 \
    --model-name model_combined_all
```

### Custom Training with All Options
```bash
python prosthetic_control/scripts/train_model.py \
    --data data/raw/prosthetic-control/20251202 \
    --model-name my_experiment_v1 \
    --test-size 0.2 \
    --cv-folds 5 \
    --scaler robust \
    --classifier-type lda \
    --window 0.2 \
    --overlap 0.1
```

### Comparing Classifiers

Train with LDA (Linear Discriminant Analysis):
```bash
python prosthetic_control/scripts/train_model.py \
    --data data/raw/prosthetic-control/20251202 \
    --model-name model_lda \
    --classifier-type lda
```

Train with SVM (Support Vector Machine):
```bash
python prosthetic_control/scripts/train_model.py \
    --data data/raw/prosthetic-control/20251202 \
    --model-name model_svm \
    --classifier-type svm
```

### Training Parameters

- `--data`: One or more data directories containing CSV files (required)
- `--model-name`: Name for saved model (default: auto-generated timestamp)
- `--model-dir`: Directory to save models (default: `models`)
- `--test-size`: Proportion for test set (default: 0.2)
- `--cv-folds`: Number of cross-validation folds (default: 5)
- `--scaler`: Scaler type - `standard`, `robust`, or `minmax` (default: `robust`)
- `--classifier-type`: Classifier - `lda` or `svm` (default: `lda`)
- `--window`: Window size in seconds (default: 0.2)
- `--overlap`: Window overlap in seconds (default: 0.1)
- `--fs-imu`: IMU sampling frequency (default: 148.1481 Hz)
- `--no-cv`: Skip cross-validation
- `--verbose`: Show detailed output

**Note:** EMG sampling frequencies (962.963 Hz for Avanti, 1259.2593 Hz for Galileo) are automatically detected from CSV headers.

---

## Real-Time Classification

### Start Real-Time System (No Motor)

For testing predictions without controlling the prosthetic hand:
```bash
python prosthetic_control/scripts/realtime_classify.py \
    --model-name model_combined_all \
    --no-motor
```

### With Motor Control

Control the prosthetic hand in real-time:
```bash
python prosthetic_control/scripts/realtime_classify.py \
    --model-name model_combined_all \
    --motor-port COM3
```

Auto-detect Arduino port:
```bash
python scripts/realtime_classify.py --model-name model_combined_all
```

### Real-Time Parameters

- `--model-name`: Name of trained model to use (required)
- `--motor-port`: Arduino serial port (e.g., `COM3` or `/dev/ttyACM0`)
- `--no-motor`: Disable motor control (testing only)
- `--window-sec`: Window size in seconds (default: 0.2)
- `--overlap-sec`: Window overlap in seconds (default: 0.1)
- `--fs-imu`: IMU sampling frequency (default: 148.1481 Hz)
- `--skip-diagnostics`: Skip channel diagnostic screen
- `--verbose`: Show buffer status and detailed info

### Prerequisites

1. **Hardware:**
   - Delsys Trigno base station connected via USB
   - Sensors powered on and paired (7 sensors: 5 Avanti, 2 Galileo)
   - Arduino Nano with servos (if using motor control)

2. **Software:**
   - Model already trained
   - Delsys credentials in `resources/delsys_key.txt` and `resources/delsys_license.lic`
   - Arduino code uploaded (if using motor control)

3. **Python packages:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Motor Control

### Test Motor Without Sensors

Cycle through gestures to test Arduino and servos:
```bash
python prosthetic_control/tests/motor_test.py --motor-port COM3
```

With custom timing:
```bash
python prosthetic_control/tests/motor_test.py --delay 2 --cycles 5
```

### Motor Test Parameters

- `--motor-port`: Arduino serial port (auto-detects if not specified)
- `--delay`: Seconds between gestures (default: 3.0)
- `--cycles`: Number of cycles (default: 0 = infinite)

### Upload Arduino Code

1. Open PlatformIO
2. Navigate to `prosthetic_control/motor/`
3. Upload `src/main.cpp` to Arduino Nano

Or via command line:
```bash
cd prosthetic_control/motor
pio run --target upload
```

### Gesture Mapping

Python sends these commands to Arduino:
- **0** = Neutral (all fingers open)
- **1** = Pinching (thumb + index)
- **2** = Grasping (full hand close)
- **3** = Zipping (clench/unclench cycle)

---

## Sensor Management

### View Paired Sensors

Check which sensors are currently paired:
```bash
python prosthetic_control/scripts/pair_sensor.py
```

Output shows:
- Currently paired sensors (Pair #, SID, Mode)
- Active sensors in range
- Option to pair new sensors

### Pair New Sensors

Follow the interactive prompts:
1. Run `python prosthetic_control/scripts/pair_sensor.py`
2. Choose option 1 to pair new sensors
3. Follow on-screen instructions

### Sensor Configuration

Your system uses:
- **5 Avanti sensors** (1-channel EMG @ 1259.26 Hz)
- **2 Galileo sensors** (4-channel EMG @ 962.96 Hz)
- All sensors include **IMU** (ACC + GYRO @ 148.15 Hz)

Total channels: ~60 (13 EMG + 21 ACC + 21 GYRO + 5 SkinCheck)

---

## Data Organization

### File Naming Convention

CSV files are named by class number:
```
<CLASS>.csv
```

Examples:
- `0.csv` - Neutral gesture
- `1.csv` - Pinching gesture
- `2.csv` - Grasping gesture
- `3.csv` - Zipping gesture

### Data Directory Structure

```
data/raw/prosthetic-control/
├── 20251113/
│   ├── 0.csv
│   ├── 1.csv
│   ├── 2.csv
│   └── 3.csv
├── 20251201/
│   ├── 0.csv
│   └── ...
└── 20260310/
    ├── 0.csv
    └── ...
```

### CSV Format

Each CSV contains:
- **Header row** with channel names and sampling frequencies
- **Timestamp column** (milliseconds)
- **EMG channels** (one or more per sensor)
- **IMU channels** (ACC X/Y/Z, GYRO X/Y/Z per sensor)

Example header:
```
timestamp,Avanti Sensor 1 - EMG 1 [1259.2593],Galileo Sensor 2 - EMG 1 [962.963],...
```

Sampling rates are automatically parsed from the header.

---

## Common Tasks

### Check Trained Models

List all available models:
```bash
ls models/
```

Each model directory contains:
- `model_bundle.pkl` - Trained classifier pipeline
- `metadata.json` - Training configuration and metrics

### Troubleshooting

**No sensors detected:**
```bash
python prosthetic_control/scripts/pair_sensor.py
```

**IMU channels missing:**
- Check that `scan_sensors()` in `delsys_client.py` enables all channels
- Verify channel breakdown shows ACC and GYRO channels

**Motor not responding:**
```bash
python prosthetic_control/tests/motor_test.py
```

**Feature count mismatch:**
- Retrain model with current sensor configuration
- Ensure same sensors are used for training and real-time

---

## Quick Reference

### Complete Workflow

1. **Pair sensors:**
   ```bash
   python prosthetic_control/scripts/pair_sensor.py
   ```

2. **Collect training data** (via Delsys software or custom script)

3. **Train model:**
   ```bash
   python prosthetic_control/scripts/train_model.py --data data/raw/prosthetic-control/20251202 --model-name model_latest
   ```

4. **Upload Arduino code:**
   ```bash
   cd prosthetic_control/motor && pio run --target upload
   ```

5. **Test motor:**
   ```bash
   python prosthetic_control/tests/motor_test.py
   ```

6. **Run real-time classification:**
   ```bash
   python prosthetic_control/scripts/realtime_classify.py --model-name model_latest
   ```

---

## Support

For issues:
1. Check that all sensors are paired and active
2. Verify Delsys credentials are correct
3. Ensure model was trained with current sensor configuration
4. Review console output for specific error messages
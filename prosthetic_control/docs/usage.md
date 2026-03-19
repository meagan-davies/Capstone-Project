# Usage Guide

## Training a Model

### Quick Start

Train a model with default settings:
```bash
python scripts/train_model.py --data-dir data/raw/20251202
```

### Custom Training
```bash
python scripts/train_model.py \
    --data-dir data/20251202 \
    --model-name my_experiment_v1 \
    --test-size 0.2 \
    --cv-folds 5 \
    --scaler robust
```

### Parameters

- `--data-dir`: Directory containing CSV files
- `--model-name`: Name for saved model (default: model_latest)
- `--test-size`: Proportion for test set (default: 0.2)
- `--cv-folds`: Number of cross-validation folds (default: 5)
- `--scaler`: Type of scaler - standard, robust, or minmax (default: standard)

## Real-Time Classification

### Start Real-Time System
```bash
python scripts/realtime_classify.py --model-name model_latest
```

### Prerequisites

1. Delsys base station connected via USB
2. Sensors powered on and paired
3. Model already trained

## Data Organization

### File Naming Convention

Files must follow this pattern:
```
CLASS.SUBCLASS_PARTICIPANT_DATE.csv
```

Examples:
- `0.1_1_20251202.csv` - Neutral, arm elevated, participant id 1
- `1.2_2_20251207.csv` - Pinching, arm at side, participant id 2

### Data Directory Structure
```
data/
├── 20251202/
│   ├── 0.1_1_20251202.csv
│   ├── 0.2_1_20251202.csv
│   ├── 1.1_1_20251202.csv
│   └── ...
└── 20251207/
    ├── 0.1_2_20251207.csv
    └── ...
```

## Common Tasks

### List Available Models
```bash
python -c "from src.models.model_utils import list_available_models; print(list_available_models())"
```

### Compare Models
```bash
python -c "from src.models.model_utils import compare_models; compare_models(['model_v1', 'model_v2'])"
```

### View Model Info
```bash
cat models/saved_models/model_latest/metadata.json
```
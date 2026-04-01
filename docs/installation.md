# Installation Guide

## Prerequisites

- Python 3.8 or higher
- Delsys Trigno system with paired sensors
- Windows 10/11 (for Delsys API support)

## Step 1: Clone Repository

```bash
git clone <https://github.com/meagan-davies/Capstone-Project.git>
```

## Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

## Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Step 4: Install Delsys API

1. Get Delsys API files from their Example-Applications repository:
   ```bash
   git clone https://github.com/delsys-inc/Example-Applications.git
   ```

2. Copy the Python API files to your project:
   ```bash
   # Copy DelsysAPI folder
   cp -r Example-Applications/Python/
   ```

3. Install pythonnet (required for Delsys API):
   ```bash
   pip install pythonnet
   ```

## Step 5: Add Delsys Credentials

1. Create credential files:
   ```bash
   # Create files
   touch resources/delsys_key.txt
   touch resources/delsys_license.lic
   ```

2. Add your credentials:
   - Open `resources/delsys_key.txt` and paste your API key
   - Open `resources/delsys_license.lic` and paste your license

   (Get these from Delsys support if you don't have them)

## Step 6: Verify Installation

Test that everything is installed correctly:

```bash
# Test Python imports
python -c "import numpy; import sklearn; print('✓ Python packages OK')"

# Test Delsys API
python src/realtime/delsys_client.py
```

## Troubleshooting

### "pythonnet not found"
```bash
pip install pythonnet
```

### "DelsysAPI.dll not found"
Make sure you copied the files to `resources/DelsysAPI/`

## Next Steps

After installation:
- See [usage.md](usage.md) for how to train models
- See main [README.md](../README.md) for project overview
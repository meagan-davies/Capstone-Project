# Capstone-Project

### Set-Up
- Create virtual environment: py -m venv venv
- Activate virtual environment: venv\Scripts\activate

### Dependencies
- pip install numpy matplotlib scikit-learn pandas seaborn

### Model Version Summary

#### LDA Models
##### Model 1.1
- Uses first 4 EMG sensors from one participant (20251113-Data)
- Classes: 
    - (0) Lifting
    - (1) Zipping 
    - (2) Pinching
- Features:
    - MAV: mean absolute value
    - RMS: root mean square
    - WL: waveform length
- Accuracy: 42%

##### Model 1.2
- Uses all EMG sensors and trained on two participants data (20251201-Data)
- Classes:
    - (0) Neutral
    - (1) Pinching
    - (2) Grasping
    - (3) Zipping
- Features:
    - MAV: mean absolute value
    - RMS: root mean square
    - WL: waveform length
- Accuracy: 65%

##### Model 1.3
- Uses all EMG sensors and trained on two participants data (20251201-Data)
- Classes:
    - (0) Neutral
    - (1) Pinching
    - (2) Grasping
    - (3) Zipping
- Features:
    - MAV: mean absolute value
    - RMS: root mean square
    - WL: waveform length
    - ZC: zero crossings
    - SSC: slope sign changes
    - VAR: variance
    - IEMG: integrated emg
- Accuracy: 70%

##### Model 1.4
- Uses all EMG and IMU sensors and trained on 3 sessions (20251201-Data, 20251202-Data)
- Classes:
    - (0) Neutral
    - (1) Pinching
    - (2) Grasping
    - (3) Zipping
- EMG Features (8 per channel):
    - MAV: mean absolute value
    - RMS: root mean square
    - VAR: variance
    - WL: waveform length
    - ZC: zero crossings
    - SSC: slope sign changes
    - MNF: mean frequency
    - MDF: median frequency
- IMU Features (25 per sensor):
    - Mean: mean value per axis (3 for acc, 3 for gyro)
    - STD: standard deviation per axis (3 for acc, 3 for gyro)
    - RMS: root mean square per axis (3 for acc, 3 for gyro)
    - Range: max-min per axis (3 for acc, 3 for gyro)
    - SMA: signal magnitude area (1 combined feature)
- Accuracy: 94%
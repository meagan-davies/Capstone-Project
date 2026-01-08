from setuptools import setup, find_packages

setup(
    name="capstone-emg-prosthetic-control",
    version="0.1.0",
    author="Meagan Davies",
    description="EMG+IMU classification for prosthetic hand control",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pyyaml>=5.4.0",
        "pythonnet>=3.0.0",
    ],
    python_requires=">=3.8",
)
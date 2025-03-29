# Workload Assessor

This repository contains the code module for real-time Mental Workload (MWL) detection from pilot control behavior using behavioral entropy. The code implements the proposed HTM-WL metric along with other established measures such as Fessonia, DNDEB, and a Naïve Forecaster. It also includes a real-time MWL spike-detector and evaluation criteria for detection lag and precision.

## Repository Structure

- **source/pipeline/run_pipeline.py**: Main script to execute the data processing pipeline.
- **configs/run_pipeline.yaml**: Configuration file for the pipeline.
- **src/**: Additional modules and helper functions.
- **requirements.txt**: Python dependencies.
- **README.md**: This file.

## Requirements

- Python 3.7 or higher.
- Required packages (see `requirements.txt`):
  - numpy
  - pandas
  - scikit-learn
  - tensorflow (or the appropriate ML library for HTM)
  - matplotlib
  - and others as needed

To install the dependencies, run:

```bash
pip install -r requirements.txt

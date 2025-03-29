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
  - (and others as needed)

To install the dependencies, run:

```bash
pip install -r requirements.txt
```


# Running the Pipeline

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/gotham29/workload_assessor.git
   cd workload_assessor
   ```

2. **Configure the Pipeline:**

   Open the configuration file at `configs/run_pipeline.yaml` and update the following fields:
   - **input_dir**: Set this to the directory containing your input data (e.g., the NASA dataset provided as supplementary material).
   - **output_dir**: Set this to the directory where you want the output results to be saved.

3. **Execute the Pipeline:**

   Run the pipeline by calling the main script with the configuration file:

   ```bash
   python source/pipeline/run_pipeline.py --config_path configs/run_pipeline.yaml
   ```


# Data Availability

The NASA dataset used in this study is provided as supplementary material with the submission. Please ensure that the dataset is placed in the directory specified by the `input_dir` parameter in `configs/run_pipeline.yaml`.


# Output

After running the pipeline, the output directory will contain:
- Processed data files.
- Plots of detection lag and precision distributions.
- Performance metrics for the evaluated MWL measures.


# Troubleshooting

- **Missing Dependencies:**  
  If you encounter errors regarding missing packages, please run:
  ```bash
  pip install -r requirements.txt
  ```
  to install all necessary dependencies.

  Path Issues:
  Verify that the paths specified in configs/run_pipeline.yaml are correct and accessible.
  Error Messages:
  Check the terminal output for any error messages; they may provide guidance on resolving issues.


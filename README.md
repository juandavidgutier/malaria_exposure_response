# Project Title

Integrating Causal Inference and Machine Learning to Quantify Climate-Malaria Relationships: Evidence of Temperature and Rainfall Thresholds from Colombian Municipalities.

## Description

Code and dataset shared to reproduce the results of the paper Integrating Causal Inference and Machine Learning to Quantify Climate-Malaria Relationships: Evidence of Temperature and Rainfall Thresholds from Colombian Municipalities.
The file dataset.csv is the dataset used for the results of the causal machine learning framework presented in the manuscript. 
To reproduce the results of the identification analysis, use the file DAG_rainfall_temperature.R.
To reproduce the results of the causal machine learning implementation, use the file exposure_response_curve.py.
To reproduce the results of the E-Value test, use the file evalue_rain_temp.

## Data Privacy and Anonymization

This dataset has been processed to ensure complete anonymization and contains no personally identifiable information (PII). All data has been:

- Aggregated at appropriate spatial/temporal scales
- Stripped of any individual identifiers
- Processed to remove direct or indirect identifying elements

The dataset is suitable for public sharing and complies with data privacy standards.

## Privacy Statement

This repository contains datasets that have been carefully processed to protect individual privacy:

### What is NOT included:
- Names, addresses, or contact information
- Individual-level identifiers
- Location data below 25 km resolution
- Timestamps more precise than monthly
- Any data that could be used to re-identify individuals

### Data Processing:
- Spatial aggregation to municipality
- Temporal aggregation to monthly averages

### Compliance:
This dataset meets requirements for public data sharing under applicable privacy regulations.

## libraries

tidyverse; dplyr; ggdag; dagitty; dplyr; GGally; tidyr; MKdescr; EValue
numpy; pandas; scipy; causal_curve; statsmodels; matplolib; plotnine; scikit-learn


## Author

Juan David Gutiérrez  


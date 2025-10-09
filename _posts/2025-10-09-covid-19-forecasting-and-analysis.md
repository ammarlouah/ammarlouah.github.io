---
title: "COVID-19 Forecasting and Risk Prediction Project"
date: 2025-10-09
categories: [Projects, Machine Learning, Deep Learning]
tags: [Projects, Forecasting, Risk Prediction, Python, TensorFlow, CatBoost, Streamlit, FastAPI, Cassandra]
---

# COVID-19 Forecasting and Risk Prediction Project

This project presents a system for forecasting COVID-19 trends and predicting patient-level risks using advanced machine learning techniques. Available on GitHub at [ammarlouah/covid-19-forecasting-and-analysis](https://github.com/ammarlouah/covid-19-forecasting-and-analysis), it integrates time-series forecasting with patient risk assessment, powered by a Temporal Fusion Transformer (TFT), CatBoost classifiers, FastAPI, Streamlit, and Cassandra. Below, I detail the project’s components, functionality, setup instructions, and key implementation insights.

## Project Overview

The project addresses two core objectives:

1. **Time-Series Forecasting**: Predicts daily new confirmed, recovered, and death counts for COVID-19 over a 7-day horizon using a pre-trained TFT model.
2. **Patient Risk Prediction**: Estimates ICU admission and mortality risks for patients based on clinical features, using CatBoost classifiers, with an interactive Streamlit dashboard for data visualization and prediction.

The system combines a FastAPI backend for predictions, a Streamlit frontend for user interaction, and a Cassandra database for efficient data storage and retrieval, creating a full-stack machine learning solution.

## Video Demonstrations

To illustrate the project’s capabilities, here are three video walkthroughs:

1. **Dashboard (Forecasting)**:
   <div class="video-container" style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden;">
       <iframe style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" src="https://www.youtube.com/embed/7U_0PrYrkIQ" frameborder="0" allowfullscreen></iframe>
   </div>

2. **Visualization (Symptom Records)**:
   <div class="video-container" style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden;">
       <iframe style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" src="https://www.youtube.com/embed/T6PqkXb80u0" frameborder="0" allowfullscreen></iframe>
   </div>

3. **Prediction (Making & Saving Patient Predictions)**:
   <div class="video-container" style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden;">
       <iframe style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" src="https://www.youtube.com/embed/RfBqJ8zmSBI" frameborder="0" allowfullscreen></iframe>
   </div>

## Repository Structure

The [GitHub repository](https://github.com/ammarlouah/covid-19-forecasting-and-analysis) is organized as follows:

- **`API/`**:
  - `main.py`: FastAPI application exposing three endpoints:
    - `POST /predict`: Generates 7-day forecasts for confirmed, recovered, and death counts from a 30x3 input sequence.
    - `POST /predict_icu`: Predicts ICU admission probability and binary outcome.
    - `POST /predict_mortality`: Predicts mortality probability and binary outcome.
  - `custom_layers.py`: Custom Keras layers for loading the TFT model.
  - `Test/`: Scripts for testing API endpoints.

- **`Dashboard/`**:
  - `dashboard.py`: Streamlit application with two pages:
    - **Dashboard**: Visualizes historical and forecasted COVID-19 data from Cassandra.
    - **Risk Prediction**: Enables exploration of patient symptom records and new predictions, saved to Cassandra.

- **`Cassandra/`**:
  - `connection.py`: Example script for connecting to a local Cassandra cluster.
  - `Forecasting/insertion.py`: Inserts time-series data from `time-series-19-covid-combined.csv`.
  - `ICU/insertion_icu.py`: Inserts patient symptom records from `covid_symptoms_severity_prediction.csv`.

- **`Dataset/Training/`**: CSV files and notebooks for dataset preparation.
- **`Model/`**:
  - `TFT/`: Contains the saved TFT model (`model.keras`) and scaler (`scaler.pkl`).
  - `ICU/`: Contains CatBoost models (`catboost_icu_admission.cbm`, `catboost_mortality.cbm`).

## Functionality

### FastAPI Backend
The backend (`API/main.py`) serves as the prediction engine, loading:
- A pre-trained TFT model for time-series forecasting.
- Two CatBoost classifiers for ICU admission and mortality predictions.

It exposes endpoints to:
- Forecast 7-day COVID-19 metrics based on 30 days of historical data.
- Predict patient-level risks from structured clinical features.

### Streamlit Dashboard
The frontend (`Dashboard/dashboard.py`) offers:
- Visualization of historical and forecasted COVID-19 data (cumulative and daily new cases) queried from Cassandra.
- A user interface to input patient features, generate risk predictions via the FastAPI endpoints, and save results to Cassandra.

### Cassandra Integration
Cassandra scripts manage data persistence:
- Store historical time-series and patient symptom data.
- Seed the database with provided CSV datasets for dashboard and API use.

## Setup Instructions

### Prerequisites
- **OS**: Windows (PowerShell recommended).
- **Python**: 3.10+ (developed with 3.11).
- **Cassandra**: Local instance running on `127.0.0.1:9042`.
- **Ports**: FastAPI uses `127.0.0.1:8000`.
- **Model Files**: Ensure models are in `Model/TFT/` and `Model/ICU/`.

### Cloning the Repository
To get started, clone the repository:
```bash
git clone https://github.com/ammarlouah/covid-19-forecasting-and-analysis.git
cd covid-19-forecasting-and-analysis
```

### Installation
1. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
   If `requirements.txt` is unavailable, install manually:
   ```powershell
   pip install fastapi uvicorn streamlit pandas numpy plotly scikit-learn catboost tensorflow joblib cassandra-driver
   ```

### Running the FastAPI Backend
From the project root:
```powershell
.\.venv\Scripts\Activate.ps1
uvicorn API.main:app --reload --host 127.0.0.1 --port 8000
```
The API loads the TFT and CatBoost models. Test endpoints using scripts in `API/Test/`.

### Launching the Streamlit Dashboard
With the API running:
```powershell
.\.venv\Scripts\Activate.ps1
streamlit run Dashboard/dashboard.py
```
Access the dashboard in your browser and navigate between the *Dashboard* and *Risk Prediction* pages.

### Seeding the Cassandra Database
Ensure Cassandra is running on `127.0.0.1:9042`, then:
- Insert time-series data:
  ```powershell
  python Cassandra\Forecasting\insertion.py
  ```
- Insert patient symptom records:
  ```powershell
  python Cassandra\ICU\insertion_icu.py
  ```

## Implementation Details

- **Forecasting Pipeline** (`API/main.py`):
  - Processes a 30x3 array of daily counts using log1p transformation and scaling (`scaler.pkl`), runs the TFT model, and inverse-transforms the output for 7-day forecasts.
- **Custom Layers** (`API/custom_layers.py`): Defines `TFTEncoderLayer` and `TFTDecoderLayer` to support loading the Keras TFT model.
- **Risk Prediction**: CatBoost models (`catboost_icu_admission.cbm`, `catboost_mortality.cbm`) predict probabilities and binary outcomes.
- **Dashboard**: Employs Streamlit caching for performance and retrieves data from Cassandra for visualization.

## Troubleshooting

- **Model Loading Errors**:
  - Verify model files exist at `Model/TFT/model.keras`, `Model/TFT/scaler.pkl`, and `Model/ICU/*.cbm`.
  - Ensure `API/custom_layers.py` is importable for TensorFlow.
- **Cassandra Issues**:
  - Confirm Cassandra is running on `127.0.0.1:9042` and check permissions.
- **Timeouts**: Adjust the 30-second timeout in `Dashboard/dashboard.py` if API calls fail.

## License

The project is released under the MIT License. See the [LICENSE file](https://github.com/ammarlouah/covid-19-forecasting-and-analysis/blob/main/LICENSE) for details.

## Contributing

Contributions are encouraged! To contribute:
- Fork the repository.
- Create a feature branch (`git checkout -b feature/my-change`).
- Commit changes with descriptive messages and push to your fork.
- Open a Pull Request to discuss your contribution.

For significant changes, please open an issue first to align on design.

## Contact

For questions or feedback, reach out via email at [ammarlouah9@gmail.com](mailto:ammarlouah9@gmail.com).

Explore the full project and try it out at [ammarlouah/covid-19-forecasting-and-analysis](https://github.com/ammarlouah/covid-19-forecasting-and-analysis)!

*Last updated: October 9, 2025*

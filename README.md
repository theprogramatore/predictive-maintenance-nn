# Predictive Maintenance Neural Network

A neural-network application developed as an educational project to estimate the probability of machine failure using machine operating data.

## Project Objective

The system uses historical machine data to train a neural network that estimates failure probability from operating conditions.

The application does not predict the exact time of failure. It provides a risk estimate that can support predictive-maintenance decisions.

## Input Variables

- Product Type
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

## Neural Network

Architecture:

`8 inputs → 32 neurons → 16 neurons → 1 output`

- Hidden layers: ReLU activation
- Output layer: Sigmoid activation
- Optimizer: Adam
- Loss function: Binary Cross-Entropy

## Data Preprocessing

- StandardScaler for numerical variables
- One-Hot Encoding for Product Type
- Training / Validation / Test split: 70% / 15% / 15%

## Model Evaluation

At the selected decision threshold of 0.60:

- Recall: 86.27%
- Precision: 27.33%
- Accuracy: 91.73%
- ROC-AUC: approximately 0.972
- PR-AUC: approximately 0.692

The model detected 44 of 51 real failures in the test set.

## Application

The user interface was developed with Streamlit.

Prediction flow:

`Machine Data → Preprocessing → Trained Neural Network → Failure Probability → Threshold → NORMAL / FAILURE WARNING`

## Online Application

The deployed application is available at:

https://predictive-maintenance-franklin.streamlit.app

## Technologies

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Joblib
- Git / GitHub

## Project Structure

- `app.py` — Streamlit user interface
- `predictor.py` — inference and prediction logic
- `preprocess.py` — preprocessing pipeline
- `build_model.py` — neural-network architecture
- `predictive_maintenance_nn.keras` — trained model
- `preprocessor.joblib` — fitted preprocessing pipeline
- `model_config.json` — model configuration and threshold

## Future Development

A future industrial version could receive real-time machine data from sensors and PLCs through technologies such as OPC UA, MQTT, or SCADA.

Additional variables such as vibration, motor current, bearing temperature, load, and operating hours could also be integrated.

---

Educational Artificial Intelligence / Predictive Maintenance Project

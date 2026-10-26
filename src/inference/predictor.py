from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import tensorflow as tf


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "predictive_maintenance_nn.keras"
)

PREPROCESSOR_PATH = (
    PROJECT_ROOT
    / "models"
    / "preprocessor.joblib"
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "model_config.json"
)


class PredictiveMaintenancePredictor:
    """
    Loads the trained neural network, fitted preprocessor,
    and model configuration for inference.
    """

    def __init__(self) -> None:
        self._validate_artifact_paths()

        # Load fitted preprocessing pipeline
        self.preprocessor = joblib.load(
            PREPROCESSOR_PATH
        )

        # Load trained neural network
        self.model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
        )

        # Load configuration
        with open(
            CONFIG_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            self.config = json.load(file)

        self.threshold = float(
            self.config["threshold"]
        )

        self.numeric_features = list(
            self.config["numeric_features"]
        )

        self.categorical_features = list(
            self.config["categorical_features"]
        )

        self.feature_columns = (
            self.numeric_features
            + self.categorical_features
        )

    def _validate_artifact_paths(self) -> None:
        """
        Verify that all required trained artifacts exist.
        """

        required_paths = [
            MODEL_PATH,
            PREPROCESSOR_PATH,
            CONFIG_PATH,
        ]

        missing = [
            str(path)
            for path in required_paths
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Missing model artifacts:\n"
                + "\n".join(missing)
            )

    def _validate_input(
        self,
        machine_data: dict[str, Any],
    ) -> None:
        """
        Validate required input fields before prediction.
        """

        missing_features = [
            feature
            for feature in self.feature_columns
            if feature not in machine_data
        ]

        if missing_features:
            raise ValueError(
                "Missing required features: "
                + ", ".join(missing_features)
            )

        valid_types = {"L", "M", "H"}

        if machine_data["Type"] not in valid_types:
            raise ValueError(
                "Type must be one of: L, M, H"
            )

        for feature in self.numeric_features:
            try:
                float(machine_data[feature])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{feature} must be numeric."
                ) from exc

    def predict(
        self,
        machine_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Predict machine failure probability
        for one machine observation.
        """

        self._validate_input(machine_data)

        # Create one-row DataFrame
        input_df = pd.DataFrame(
            [machine_data],
            columns=self.feature_columns,
        )

        # Apply fitted preprocessing
        processed_data = (
            self.preprocessor.transform(input_df)
        )

        # Neural-network probability
        probability = float(
            self.model.predict(
                processed_data,
                verbose=0,
            )[0][0]
        )

        predicted_failure = (
            probability >= self.threshold
        )

        status = (
            "FAILURE WARNING"
            if predicted_failure
            else "NORMAL"
        )

        return {
            "probability": probability,
            "probability_percent": probability * 100,
            "threshold": self.threshold,
            "predicted_class": int(
                predicted_failure
            ),
            "status": status,
        }
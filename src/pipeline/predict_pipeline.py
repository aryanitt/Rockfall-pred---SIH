import sys
import pandas as pd
from src.exception_config import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        try:
            # Load preprocessor, model, and training feature columns
            self.model = load_object("artifacts/model.pkl")
            self.preprocessor = load_object("artifacts/preprocessor.pkl")
            self.feature_columns = load_object("artifacts/feature_columns.pkl")  # list of column names used during training
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features: pd.DataFrame):
        try:
            # Add missing columns with default value 0
            for col in self.feature_columns:
                if col not in features.columns:
                    features[col] = 0

            # Reorder columns to match training
            features = features[self.feature_columns]

            # Transform and predict
            data_scaled = self.preprocessor.transform(features)
            preds = self.model.predict(data_scaled)

            return preds  # 0 = No Risk, 1 = Risk

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 slope_angle: float,
                 rock_density: float,
                 vibration: float,
                 rainfall: float,
                 temperature: float,
                 soil_moisture: float,
                 joint_spacing: float,
                 weathering: str):
        """
        Custom data object for rockfall prediction.
        """
        self.slope_angle = slope_angle
        self.rock_density = rock_density
        self.vibration = vibration
        self.rainfall = rainfall
        self.temperature = temperature
        self.soil_moisture = soil_moisture
        self.joint_spacing = joint_spacing
        self.weathering = weathering  # categorical feature (e.g., low, medium, high)

    def get_data_as_dataframe(self):
        try:
            data_dict = {
                "slope_angle": [self.slope_angle],
                "rock_density": [self.rock_density],
                "vibration": [self.vibration],
                "rainfall": [self.rainfall],
                "temperature": [self.temperature],
                "soil_moisture": [self.soil_moisture],
                "joint_spacing": [self.joint_spacing],
                "weathering": [self.weathering],
            }

            return pd.DataFrame(data_dict)

        except Exception as e:
            raise CustomException(e, sys)

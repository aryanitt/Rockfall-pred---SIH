import os
import sys
import pandas as pd
import numpy as np
from src.logger_config import logging
from src.exception_config import CustomException

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    feature_columns_file_path: str = os.path.join('artifacts', 'feature_columns.pkl')


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        """
        Returns the preprocessor for numerical features.
        """
        try:
            numerical_features = [
                'slope', 'elevation', 'rainfall_7d', 'temperature',
                'vibration_mean', 'vibration_max', 'spikes'
            ]

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            logging.info("Numerical pipeline created for features: {}".format(numerical_features))

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # 1️⃣ Load datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data read successfully")

            # 2️⃣ Get preprocessing object
            preprocessor = self.get_data_transformation_object()

            target_column = "risk"

            # 3️⃣ Separate features and target
            X_train = train_df.drop(columns=[target_column], axis=1)
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column], axis=1)
            y_test = test_df[target_column]

            # 4️⃣ Fit and transform
            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)
            logging.info(f"Transformed train shape: {X_train_arr.shape}, test shape: {X_test_arr.shape}")

            # 5️⃣ Combine features and target
            train_arr = np.c_[X_train_arr, y_train.to_numpy()]
            test_arr = np.c_[X_test_arr, y_test.to_numpy()]

            # 6️⃣ Save preprocessor
            save_object(file_path=self.transformation_config.preprocessor_file_path,
                        obj=preprocessor)
            logging.info(f"Preprocessor saved at {self.transformation_config.preprocessor_file_path}")

            # 7️⃣ Save feature columns
            feature_columns = X_train.columns.tolist()
            save_object(file_path=self.transformation_config.feature_columns_file_path,
                        obj=feature_columns)
            logging.info(f"Feature columns saved at {self.transformation_config.feature_columns_file_path}")

            return train_arr, test_arr, preprocessor

        except Exception as e:
            raise CustomException(e, sys)

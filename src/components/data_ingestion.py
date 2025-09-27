import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception_config import CustomException
from src.logger_config import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils import save_object


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    feature_columns_path: str = os.path.join('artifacts', 'feature_columns.pkl')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method/component")

        try:
            # ✅ Load dataset
            dataset_path = 'notebook/rockfall_dataset_v2.csv'
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset not found at {dataset_path}")

            df = pd.read_csv(dataset_path)
            logging.info(f"Read dataset with shape {df.shape}")

            # ✅ Verify target column
            if "risk" not in df.columns:
                raise ValueError("Target column 'risk' not found in dataset")

            # ✅ Save raw dataset
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Saved raw dataset to {self.ingestion_config.raw_data_path}")

            # ✅ Train-test split
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42, stratify=df["risk"]
            )
            logging.info(f"Train shape: {train_set.shape}, Test shape: {test_set.shape}")

            # Save train/test splits
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion complete")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        # 1️⃣ Data ingestion
        obj = DataIngestion()
        train_path, test_path = obj.initiate_data_ingestion()

        # 2️⃣ Data transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor = data_transformation.initiate_data_transformation(train_path, test_path)

        # 3️⃣ Save feature columns for prediction pipeline
        import pandas as pd
        train_df = pd.read_csv(train_path)
        feature_columns = train_df.drop(columns=["risk"]).columns.tolist()
        save_object("artifacts/feature_columns.pkl", feature_columns)
        logging.info(f"Saved feature columns: {feature_columns}")

        # 4️⃣ Model training
        trainer = ModelTrainer()
        result = trainer.initiate_model_trainer(train_arr, test_arr)
        print(result)

    except Exception as e:
        raise CustomException(e, sys)

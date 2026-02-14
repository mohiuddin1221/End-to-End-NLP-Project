import pandas as pd
from pathlib import Path
import logging
from config import Fake_data, True_Data
logger = logging.getLogger(__name__)




def ingest_data(fake_Data: str = Fake_data, true_Data: str = True_Data) -> pd.DataFrame:
    fake_data_path = Path(Fake_data)
    true_data_apth = Path(True_Data)

    if not fake_data_path.exists():
        logger.error(f"Data source not found at {fake_data_path}")
        raise FileNotFoundError(f"Data source not found at {fake_data_path}")

    logger.info(f"Loading data from {fake_data_path}")

    if not true_data_apth.exists():
        logger.error(f"Data source not found at {true_data_apth}")
        raise FileNotFoundError(f"Data source not found at {true_data_apth}")

    logger.info(f"Loading data from {true_data_apth}")

    try:
        fake_data = pd.read_csv(fake_Data)
        fake_data["label"] = "fake"
        logger.info(f"Data loaded successfully with shape {fake_data.shape}")

        true_Data = pd.read_csv(true_Data)
        true_Data["label"] = "true"
        logger.info(f"Data loaded successfully with shape {true_Data.shape}")

        data = pd.concat([true_Data, fake_data], axis=0)
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)
        data["title_len"] = data["title"].str.len()
        data["text_len"] = data["text"].str.len()

        data["title_word_count"] = data["title"].str.split().str.len()
        data["text_word_count"] = data["text"].str.split().str.len()
        return data

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise



import pandas as pd
from error_logs import configure_logger
logger = configure_logger()

def features_selection(data: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info("feature selection running...........................")
        unique_subject_count = data["subject"].nunique()
        print("Unique Subject Count:", unique_subject_count)

        data.drop(columns=["subject", "date"], inplace=True)
        logger.info("Remove Subject and data Colum")
        return data

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
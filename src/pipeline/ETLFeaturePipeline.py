from config import Fake_data, True_Data

from src.step.a_ingest import ingest_data
from src.step.b_features_selection import features_selection
from src.step.c_model_training import model_traning


from error_logs import configure_logger
logger = configure_logger()

def run_pipleline():
    """
    Run the end-to-end ETL and feature pipeline for the ML project.
    """
    try:
        logger.info("Starting ETL and Feature Pipeline.")

        data = ingest_data(fake_Data = Fake_data, true_Data = True_Data)
        if data is None or data.empty:
            logger.error("Ingested data is empty. Exiting pipeline.")
            return
        # Featured Data

        featured_data = features_selection(data)

        #Model Traning
        model = model_traning(featured_data)
        logger.info("ETL and Feature Pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Error in ETL and Feature Pipeline: {e}") 
        return e
    

if __name__ == "__main__":
    run_pipeline()



# python -m src.pipeline.ETLFeaturePipeline
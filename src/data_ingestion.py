import os
import kagglehub
import shutil
import zipfile # Keep zipfile import in case you add more generic zip handling later
from src.logger import get_logger
from src.custom_exception import CustomException # Assuming src/custom_exception.py exists
# Recommended: Import specific variables instead of using *
# from config.data_ingestion_config import DATASET_NAME, TARGET_DIR
import traceback # Import traceback for detailed error logging

logger = get_logger(__name__)
# Assuming these variables are defined in config/data_ingestion_config.py
# If not, you'll need to define them or import them explicitly.
try:
    from config.data_ingestion_config import DATASET_NAME, TARGET_DIR
    logger.info("Successfully imported DATASET_NAME and TARGET_DIR from config.")
except ImportError:
    logger.error("Failed to import DATASET_NAME or TARGET_DIR from config. Please ensure config/data_ingestion_config.py exists and defines these variables.")
    # Define placeholders or raise an error to prevent script execution
    DATASET_NAME = None
    TARGET_DIR = None
    # raise CustomException("Configuration not loaded.") # Uncomment to stop execution on config error

class DataIngestion:
    """
    Handles downloading and organizing dataset from Kaggle Hub.
    Assumes the Kaggle dataset, once downloaded by kagglehub, contains
    'Images' and 'Labels' subdirectories at its top level.
    """
    def __init__(self, dataset_name: str, target_dir: str):
        if not dataset_name or not target_dir:
             error_msg = "Dataset name or target directory is not set."
             logger.error(error_msg)
             raise CustomException(error_msg)

        self.dataset_name = dataset_name
        self.target_dir = target_dir
        # Define the expected raw directory path
        self.raw_dir = os.path.join(self.target_dir, "raw")

    def create_raw_dir(self):
        """
        Creates the target raw data directory if it doesn't exist.
        """
        logger.info(f"Attempting to create raw directory: {self.raw_dir}")
        if not os.path.exists(self.raw_dir):
            try:
                os.makedirs(self.raw_dir, exist_ok=True) # Use exist_ok=True for robustness
                logger.info(f"Created raw directory: {self.raw_dir}")
            except Exception as e:
                logger.error(f"Error while creating raw directory {self.raw_dir}: {e}\n{traceback.format_exc()}", exc_info=False)
                raise CustomException(f"Failed to create raw directory: {self.raw_dir}", e) from e
        else:
            logger.info(f"Raw directory already exists: {self.raw_dir}")

        return self.raw_dir

    def organize_downloaded_data(self, downloaded_path: str):
        """
        Organizes the downloaded dataset by moving 'Images' and 'Labels'
        subfolders from the downloaded location to the raw directory.

        Args:
            downloaded_path (str): The path to the directory where kagglehub
                                   downloaded and extracted the dataset.
        """
        logger.info(f"Organizing data from downloaded path: {downloaded_path}")
        images_source = os.path.join(downloaded_path, "Images")
        labels_source = os.path.join(downloaded_path, "Labels")

        images_target = os.path.join(self.raw_dir, "Images")
        labels_target = os.path.join(self.raw_dir, "Labels")

        try:
            # Move Images folder
            if os.path.exists(images_source):
                logger.info(f"Moving Images from {images_source} to {images_target}")
                # Use copytree instead of move if you want to keep the original download
                shutil.move(images_source, images_target)
                logger.info("Images folder moved successfully.")
            else:
                logger.warning(f"Images folder not found in downloaded data at {images_source}.")

            # Move Labels folder
            if os.path.exists(labels_source):
                logger.info(f"Moving Labels from {labels_source} to {labels_target}")
                 # Use copytree instead of move if you want to keep the original download
                shutil.move(labels_source, labels_target)
                logger.info("Labels folder moved successfully.")
            else:
                # This might be expected if the dataset only contains images and no labels
                logger.warning(f"Labels folder not found in downloaded data at {labels_source}. Proceeding without labels for this source.")

        except Exception as e:
            logger.error(f"Error while organizing downloaded data: {e}\n{traceback.format_exc()}", exc_info=False)
            raise CustomException("Error while organizing downloaded data", e) from e

    def download_dataset(self):
        """
        Downloads the dataset from Kaggle Hub using kagglehub.
        """
        logger.info(f"Attempting to download dataset: {self.dataset_name} from Kaggle Hub")
        try:
            # kagglehub.dataset_download downloads and extracts the dataset
            # It returns the path to the directory where the data is located
            downloaded_path = kagglehub.dataset_download(self.dataset_name)
            logger.info(f"Dataset downloaded and extracted by kagglehub to: {downloaded_path}")

            # Now organize the downloaded data into the raw directory
            self.organize_downloaded_data(downloaded_path)

        except Exception as e:
            # kagglehub.dataset_download might raise various exceptions (auth, network, etc.)
            logger.error(f"Error while downloading dataset {self.dataset_name}: {e}\n{traceback.format_exc()}", exc_info=False)
            raise CustomException(f"Error while downloading dataset {self.dataset_name}", e) from e

    def is_data_already_ingested(self):
        """
        Checks if the data appears to be already ingested in the raw directory.
        This is a simple check based on the existence of Images and Labels subfolders.
        """
        images_path = os.path.join(self.raw_dir, "Images")
        labels_path = os.path.join(self.raw_dir, "Labels")

        # Consider data ingested if both Images and Labels folders exist and are not empty
        # Or adjust this logic based on your dataset (e.g., maybe Labels is optional)
        if os.path.exists(images_path) and os.path.exists(labels_path) and os.listdir(images_path) and os.listdir(labels_path):
             logger.info(f"Data appears to be already ingested in {self.raw_dir} (Images and Labels found).")
             return True
        elif os.path.exists(images_path) and os.path.exists(images_path):
             # Handle case where folders exist but might be empty from a previous failed run
             logger.warning(f"Images or Labels directory in {self.raw_dir} is empty. Attempting re-ingestion.")
             return False # Re-ingest if directories are empty
        elif os.path.exists(self.raw_dir) and os.listdir(self.raw_dir):
             logger.warning(f"Raw directory {self.raw_dir} exists but does not contain both Images and Labels subfolders. Attempting re-ingestion.")
             return False # Re-ingest if raw_dir exists but expected folders are missing
        else:
            logger.info(f"Data does not appear to be ingested in {self.raw_dir}.")
            return False


    def run(self):
        """
        Runs the data ingestion pipeline: creates directory, downloads, and organizes data.
        Checks if data is already ingested before downloading.
        """
        logger.info("Starting data ingestion pipeline.")
        try:
            self.create_raw_dir()

            if self.is_data_already_ingested():
                logger.info("Skipping download as data is already ingested.")
            else:
                self.download_dataset()
                logger.info("Data ingestion completed.")

        except Exception as e:
            # This catches exceptions raised by create_raw_dir or download_dataset/organize_downloaded_data
            logger.error(f"Error during data ingestion pipeline: {e}\n{traceback.format_exc()}", exc_info=False)
            # The inner methods already raised CustomException, so this might be redundant
            # depending on how you want to stack exceptions, but it's safe.
            raise CustomException("Error during data ingestion pipeline", e) from e


# Example Usage:
if __name__ == "__main__":
    # Ensure DATASET_NAME and TARGET_DIR are defined in config/data_ingestion_config.py
    # Example placeholder values if config is missing (remove or update)
    # DATASET_NAME = "INSERT_YOUR_KAGGLE_DATASET_NAME_HERE" # e.g., "user/dataset-name"
    # TARGET_DIR = "artifacts/data_ingestion" # Example target directory

    if DATASET_NAME is None or TARGET_DIR is None:
         print("Configuration variables DATASET_NAME or TARGET_DIR are not set. Cannot run data ingestion.")
         exit() # Exit if config failed to load

    try:
        # Instantiate the DataIngestion class
        data_ingestion = DataIngestion(dataset_name=DATASET_NAME, target_dir=TARGET_DIR)

        # Run the ingestion pipeline
        data_ingestion.run()

        print("\nData ingestion script finished running.")
        # Add checks here to verify if the raw directory was populated as expected
        images_check_path = os.path.join(data_ingestion.raw_dir, "Images")
        labels_check_path = os.path.join(data_ingestion.raw_dir, "Labels")

        print(f"Checking for Images directory: {images_check_path}")
        print(f"Images directory exists: {os.path.exists(images_check_path)}")
        if os.path.exists(images_check_path):
            print(f"Number of items in Images directory: {len(os.listdir(images_check_path))}")

        print(f"Checking for Labels directory: {labels_check_path}")
        print(f"Labels directory exists: {os.path.exists(labels_check_path)}")
        if os.path.exists(labels_check_path):
             print(f"Number of items in Labels directory: {len(os.listdir(labels_check_path))}")


    except CustomException as ce:
        print(f"\nData Ingestion failed with CustomException: {ce}")
        # The logger would have already printed detailed error info
    except Exception as e:
         print(f"\nData Ingestion failed with an unexpected error: {e}")
         logger.error(f"An unexpected error occurred in the main execution block: {e}\n{traceback.format_exc()}", exc_info=False)
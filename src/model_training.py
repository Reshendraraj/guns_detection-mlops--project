import os
import torch
from torch import optim
from src.model_architecture import FasterRCNNModel
from src.logger import get_logger
from src.custom_exception import CustomException
from torch.utils.tensorboard import SummaryWriter
from src.model_architecture import FasterRCNNModel


import time

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, model_class, num_classes, dataset_path, device, pretrained_weights_path):
        self.model_class = model_class
        self.num_classes = num_classes
        self.dataset_path = dataset_path
        self.device = device
        self.pretrained_weights_path = pretrained_weights_path

        # TensorBoard setup
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_dir = f"tensorboard_logs/{timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # Prepare model
        try:
            self.model = self.model_class(self.num_classes, self.device).model
            self.model.load_state_dict(torch.load(self.pretrained_weights_path, map_location=self.device))
            self.model.to(self.device)
            logger.info("Pretrained model loaded and moved to device.")

        except Exception as e:
            logger.error(f"Failed to load pretrained model {e}")
            raise CustomException("Failed to load pretrained model", e)

    def save_model(self, save_dir="artifacts/models/", filename="fasterrcnn.pth"):
        try:
            os.makedirs(save_dir, exist_ok=True)
            model_save_path = os.path.join(save_dir, filename)
            torch.save(self.model.state_dict(), model_save_path)
            logger.info(f"Model saved successfully to {model_save_path}")
        except Exception as e:
            logger.error(f"Failed to save model {e}")
            raise CustomException("Failed to save model", e)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Replace this with your actual model path
    pretrained_weights_path = r"C:/Users/konde\Downloads/fasterrcnn_trained.pth"

    training = ModelTraining(
        model_class=FasterRCNNModel,
        num_classes=2,
        dataset_path="artifacts/raw/",  # Not used here but kept for compatibility
        device=device,
        pretrained_weights_path=pretrained_weights_path
    )

    training.save_model()  # Save it inside artifacts/models/

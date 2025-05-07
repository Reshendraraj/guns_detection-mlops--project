# src/models/faster_rcnn_model.py

import torch
from torch import nn
from torch.optim import Adam
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.optim.lr_scheduler import StepLR
from src.logger import get_logger
from src.custom_exception import CustomException
from tqdm import tqdm
import os

logger = get_logger(__name__)

class FasterRCNNModel:
    def __init__(self, num_classes: int, device: str = "cpu"):
        """
        Initialize the Faster R-CNN model for gun detection.
        Args:
            num_classes (int): Number of output classes including background.
            device (str): Device to train on (e.g., 'cuda' or 'cpu').
        """
        self.num_classes = num_classes
        self.device = device
        self.model = self._create_model().to(self.device)
        self.optimizer = None
        self.scheduler = None
        logger.info("FasterRCNN model initialized.")

    def _create_model(self):
        """
        Load a pre-trained Faster R-CNN and replace the classifier head.
        """
        try:
            model = fasterrcnn_resnet50_fpn(pretrained=True)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
            return model
        except Exception as e:
            logger.error("Error creating model architecture.")
            raise CustomException(e)

    def compile(self, lr=1e-4, step_size=5, gamma=0.1):
        """
        Prepare optimizer and LR scheduler.
        """
        try:
            self.optimizer = Adam(self.model.parameters(), lr=lr)
            self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)
            logger.info("Model compiled successfully with Adam optimizer and StepLR scheduler.")
        except Exception as e:
            logger.error("Error during model compilation.")
            raise CustomException(e)

    def train(self, train_loader, num_epochs=10):
        """
        Train the model using the provided DataLoader.
        """
        try:
            self.model.train()
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                logger.info(f"Epoch {epoch+1}/{num_epochs} started...")

                for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                    images = [img.to(self.device) for img in images]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                    loss_dict = self.model(images, targets)
                    total_loss = sum(loss for loss in loss_dict.values())

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()

                    epoch_loss += total_loss.item()

                self.scheduler.step()
                logger.info(f"Epoch {epoch+1} completed. Loss: {epoch_loss:.4f}")
        except Exception as e:
            logger.error("Training failed.")
            raise CustomException(e)

    def save_model(self, path: str):
        """
        Save model weights to disk.
        """
        try:
            torch.save(self.model.state_dict(), path)
            logger.info(f"Model saved at {path}")
        except Exception as e:
            logger.error("Failed to save model.")
            raise CustomException(e)

    def load_model(self, path: str):
        """
        Load model weights from disk.
        """
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error("Failed to load model.")
            raise CustomException(e)


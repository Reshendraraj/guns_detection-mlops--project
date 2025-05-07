import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class GunDataset(Dataset):
    def __init__(self, root: str, device: str = "cpu"):
        """
        Custom PyTorch dataset for object detection of guns.

        Args:
            root (str): Root folder containing 'Images/' and 'Labels/' subfolders.
            device (str): Device to send tensors to (default: 'cpu').
        """
        try:
            self.image_dir = os.path.join(root, "Images")
            self.label_dir = os.path.join(root, "Labels")
            self.device = device

            self.image_files = sorted(os.listdir(self.image_dir))
            if not self.image_files:
                raise FileNotFoundError(f"No images found in: {self.image_dir}")

            logger.info(f"GunDataset initialized with {len(self.image_files)} samples.")

        except Exception as e:
            logger.error("Dataset initialization failed.")
            raise CustomException("Error initializing GunDataset", e)

    def __getitem__(self, idx):
        try:
            img_file = self.image_files[idx]
            img_path = os.path.join(self.image_dir, img_file)

            # Load image
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Image not found or unreadable: {img_path}")

            # Convert image to RGB and normalize to [0, 1]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            image_tensor = torch.tensor(image).permute(2, 0, 1).to(self.device)

            # Load label
            label_file = os.path.splitext(img_file)[0] + ".txt"
            label_path = os.path.join(self.label_dir, label_file)
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Label file missing: {label_path}")

            with open(label_path, "r") as f:
                line_count = int(f.readline().strip())
                boxes = [list(map(int, f.readline().strip().split())) for _ in range(line_count)]

            # Initialize default target
            target = {
                "boxes": torch.tensor([], dtype=torch.float32).to(self.device),
                "labels": torch.tensor([], dtype=torch.int64).to(self.device),
                "area": torch.tensor([], dtype=torch.float32).to(self.device),
                "image_id": torch.tensor([idx]).to(self.device)
            }

            if boxes:
                # Convert boxes to tensor
                boxes_tensor = torch.tensor(boxes, dtype=torch.float32).to(self.device)

                # Compute the area of the bounding boxes
                area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])

                # Assign label 1 (you can extend this for multi-class scenarios)
                labels = torch.ones((len(boxes),), dtype=torch.int64).to(self.device)

                target.update({
                    "boxes": boxes_tensor,
                    "labels": labels,
                    "area": area
                })

            logger.debug(f"Loaded sample {idx}: {img_file}")
            return image_tensor, target

        except Exception as e:
            logger.error(f"Error loading sample index {idx}: {e}")
            raise CustomException(f"Failed to load sample {idx}", e)

    def __len__(self):
        return len(self.image_files)


# Test example (optional for debugging)
if __name__ == "__main__":
    try:
        root_dir = "artifacts/raw"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dataset = GunDataset(root=root_dir, device=device)

        image, target = dataset[0]
        print("Image shape:", image.shape)
        print("Target keys:", target.keys())
        print("Bounding Boxes:", target["boxes"])

    except Exception as e:
        logger.error(f"Test failed: {e}")

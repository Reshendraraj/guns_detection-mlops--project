import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw
# random import seems unused, can remove unless intended for future use
# If you have a logger configured at src/logger.py, ensure it's imported like this:
# from src.logger import get_logger
# logger = get_logger(__name__)
# If not, you can use Python's built-in logging module or print statements for simplicity for now.
# For this example, let's use basic print statements for immediate feedback if logger is not set up.
import logging
# Configure basic logging if src.logger is not available
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Model Loading (This happens only once when the FastAPI application starts) ---
# Loading the model here at the module level ensures it's loaded into memory
# before the first request is processed, preventing slow startup times per request.
try:
    # Load the pre-trained Faster R-CNN model
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval() # Set the model to evaluation mode - crucial for inference

    logger.info("Successfully loaded pre-trained Faster R-CNN model.")

except Exception as e:
    logger.error(f"Error loading model at application startup: {e}", exc_info=True)
    # In a production API, failing fast if the model cannot be loaded is often desired.
    # You might want to raise an exception here to prevent the app from starting.
    # raise RuntimeError("Failed to load model for the API.") from e


# Determine the device to use for model inference (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device) # Move the model to the selected device
logger.info(f"Model moved to device: {device}")


# Define the image transformation pipeline required by the model
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Create the FastAPI application instance
app = FastAPI(
    title="Guns Object Detection API",
    description="API for performing object detection on images and returning annotated results."
)


# --- Helper function for prediction and drawing (This is a blocking/CPU-bound task) ---
def perform_inference_and_draw(image: Image.Image, score_threshold: float = 0.7) -> Image.Image:
    """
    Performs object detection inference using the loaded model and draws
    bounding boxes on the input PIL Image based on a score threshold.
    This function contains blocking/CPU-bound operations.
    """
    logger.info(f"Starting inference and drawing with score threshold: {score_threshold}")
    try:
        if image.mode != "RGB":
            image_rgb = image.convert("RGB")
        else:
            image_rgb = image

        img_tensor = transform(image_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            predictions = model(img_tensor)

        prediction = predictions[0]
        boxes = prediction['boxes'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()

        logger.info(f"Inference completed. Found {len(boxes)} potential objects.")

        output_image_with_boxes = image_rgb.copy()
        draw = ImageDraw.Draw(output_image_with_boxes)

        drawn_boxes_count = 0
        for box, score in zip(boxes, scores):
            if score > score_threshold:
                x_min, y_min, x_max, y_max = box.astype(int)
                draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
                drawn_boxes_count += 1

        logger.info(f"Drawn {drawn_boxes_count} bounding boxes with score > {score_threshold}")

        return output_image_with_boxes

    except Exception as e:
        logger.error(f"Error during inference or drawing process: {e}", exc_info=True)
        # You might want a more specific custom exception here if needed
        raise Exception("Object detection inference and drawing failed.") from e


# --- API Endpoints ---

@app.get("/")
def read_root():
    """Basic root endpoint confirming the API is running."""
    return {"message": "Welcome to the Guns Object Detection API. Use the /predict/ endpoint to upload images."}


@app.post("/predict/")
async def predict(file: UploadFile = File(..., description="Image file to perform object detection on.")):
    """
    Receives an image file, performs object detection using the loaded model,
    draws bounding boxes on the image, and returns the annotated image as a PNG stream.
    """
    logger.info(f"Received request for prediction. File: {file.filename}, Content-Type: {file.content_type}")

    try:
        image_data = await file.read()
        if not image_data:
            logger.warning(f"Uploaded file {file.filename} is empty.")
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        try:
            image = Image.open(io.BytesIO(image_data))
            logger.info(f"Successfully read and opened image file: {file.filename}")
        except Exception as img_e:
            logger.error(f"Error opening image file {file.filename}: {img_e}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Could not open image file. Please ensure it is a valid image format: {img_e}")

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"An unexpected error occurred during file reading or opening for {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred while processing the file: {e}")


    logger.info(f"Offloading inference and drawing to thread pool for {file.filename}")
    try:
        output_image = await run_in_threadpool(perform_inference_and_draw, image, 0.7)
        logger.info(f"Inference and drawing completed in thread pool for {file.filename}.")

    except Exception as e: # Catching generic Exception here as perform_inference_and_draw now raises generic Exception
         logger.error(f"An error occurred during inference processing for {file.filename}: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail=f"An internal error occurred during inference: {e}")


    try:
        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        logger.info(f"Prepared output image stream for {file.filename}")

        return StreamingResponse(img_byte_arr, media_type="image/png")

    except Exception as e:
         logger.error(f"Error preparing output image stream for {file.filename}: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail=f"Error preparing image response: {e}")

# --- How to Run the FastAPI Application ---
# This part is for running the script as a server.
# You would typically use a command like:
# uvicorn src.app:app --reload
# from your project's root directory (E:\my_project_cleaned>)

# You can keep this __main__ block for easy local testing during development
if __name__ == "__main__":
    import uvicorn
    # Running with reload=True is good for development
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
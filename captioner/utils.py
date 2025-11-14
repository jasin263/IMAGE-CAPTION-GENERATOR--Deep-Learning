from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os
import logging

logger = logging.getLogger('captioner')

# Load the BLIP model and processor lazily to avoid loading at import time
processor = None
model = None

def load_model(use_trained=False, model_path="trained_model"):
    """
    Load BLIP model - either pre-trained or custom trained

    Args:
        use_trained (bool): Whether to use custom trained model
        model_path (str): Path to trained model directory
    """
    global processor, model
    if processor is None or model is None:
        try:
            if use_trained and os.path.exists(model_path):
                logger.info(f"Loading trained model from {model_path}")
                processor = BlipProcessor.from_pretrained(model_path)
                model = BlipForConditionalGeneration.from_pretrained(model_path)
                logger.info("Custom trained model loaded successfully")
            else:
                logger.info("Loading pre-trained BLIP model")
                processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                logger.info("Pre-trained BLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

def generate_caption(image_path, use_trained=False):
    """
    Generate a caption for the given image using BLIP model.

    Args:
        image_path (str): Path to the image file.
        use_trained (bool): Whether to use custom trained model.

    Returns:
        str: Generated caption for the image.
    """
    try:
        logger.info(f"Starting caption generation for image: {os.path.basename(image_path)}")

        # Load model if not already loaded
        load_model(use_trained=use_trained)

        # Open the image
        image = Image.open(image_path).convert('RGB')
        logger.debug(f"Image opened successfully: {image.size}")

        # Process the image
        inputs = processor(image, return_tensors="pt")

        # Generate caption
        with torch.no_grad():
            out = model.generate(**inputs, max_length=50, num_beams=5, early_stopping=True)

        # Decode the generated caption
        caption = processor.decode(out[0], skip_special_tokens=True)

        logger.info(f"Caption generated: '{caption}'")
        return caption
    except Exception as e:
        logger.error(f"Error generating caption: {str(e)}")
        return f"Error generating caption: {str(e)}"

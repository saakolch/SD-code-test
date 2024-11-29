import json
import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_checkpoint(url, local_path):
    if os.path.exists(local_path):
        logger.info(f"Checkpoint already exists at {local_path}")
        return local_path

    logger.info(f"Downloading checkpoint from {url} to {local_path}")
    response = requests.get(url)
    with open(local_path, 'wb') as file:
        file.write(response.content)
    return local_path

def load_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)

def load_checkpoint(ckpt_name):
    try:
        model = StableDiffusionPipeline.from_pretrained(ckpt_name)
        tokenizer = CLIPTokenizer.from_pretrained(ckpt_name)
        clip = CLIPTextModel.from_pretrained(ckpt_name)
        return model, tokenizer, clip
    except Exception as e:
        logger.error(f"Failed to load checkpoint {ckpt_name}: {e}")
        sys.exit(1)

def encode_text(text, tokenizer, clip):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = clip(**inputs)
    return outputs.last_hidden_state

def generate_latent_image(width, height, batch_size):
    return torch.randn(batch_size, 4, height // 8, width // 8)

def save_images(images, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    for idx, image in enumerate(images):
        image.save(os.path.join(output_dir, f"{prefix}_{idx}.png"))
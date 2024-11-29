import random
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import os
import json
import logging
import sys
from utils import load_config, load_checkpoint, encode_text, generate_latent_image, save_images, download_checkpoint

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(config_path):
    config = load_config(config_path)

    base_model_url = os.getenv('BASE_MODEL_URL', config['base_model_url'])
    refiner_model_url = os.getenv('REFINER_MODEL_URL', config['refiner_model_url'])

    base_model_path = download_checkpoint(base_model_url, "base_model.ckpt")
    refiner_model_path = download_checkpoint(refiner_model_url, "refiner_model.ckpt")

    with torch.inference_mode():
        # Load checkpoints
        base_model, base_tokenizer, base_clip = load_checkpoint(base_model_path)
        refiner_model, refiner_tokenizer, refiner_clip = load_checkpoint(refiner_model_path)

        # Encode text
        positive_prompt = "beautiful scenery nature glass bottle landscape, , purple galaxy bottle,"
        negative_prompt = "text, watermark"

        positive_base = encode_text(positive_prompt, base_tokenizer, base_clip)
        negative_base = encode_text(negative_prompt, base_tokenizer, base_clip)

        positive_refiner = encode_text(positive_prompt, refiner_tokenizer, refiner_clip)
        negative_refiner = encode_text(negative_prompt, refiner_tokenizer, refiner_clip)

        # Generate latent image
        latent_image = generate_latent_image(config['image_width'], config['image_height'], config['batch_size'])

        # Initialize scheduler
        scheduler = EulerDiscreteScheduler.from_pretrained(base_model_path, subfolder="scheduler")

        # Generate images
        for q in range(config['num_images']):
            # First stage sampling
            noise_seed = random.randint(1, 2**64)
            generator = torch.Generator().manual_seed(noise_seed)
            base_output = base_model(
                prompt_embeds=positive_base,
                negative_prompt_embeds=negative_base,
                latents=latent_image,
                num_inference_steps=config['steps'],
                guidance_scale=config['cfg'],
                generator=generator,
                output_type="latent"
            ).images

            # Second stage sampling
            refiner_output = refiner_model(
                prompt_embeds=positive_refiner,
                negative_prompt_embeds=negative_refiner,
                latents=base_output,
                num_inference_steps=config['steps'],
                guidance_scale=config['cfg'],
                generator=generator,
                output_type="pil"
            ).images

            # Save images
            save_images(refiner_output, config['output_dir'], f"output_{q}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python main.py <config_path>")
        sys.exit(1)
    main(sys.argv[1])


import torch
from diffusers import StableDiffusionPipeline
from google.colab import files
import os
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import zipfile

# Check if CUDA is available and set device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)
pipe.safety_checker = None

prompt = "Beautiful Woman, 20 Years old, piercing green eyes, long curly brown hair, flawless skin, wet body dripping, intimate sexy pose, photo realistic skin texture, natural lighting, golden hour lighting, dramatic lighting, cinematic lighting, backlighting, volume metric lighting, lens flare, motion blur, cgi, depths of field, tilt shift, prime lens, f/1.4, Beautiful Woman, 20 Years old, piercing green eyes, long curly brown hair, flawless skin, wet body dripping, intimate sexy pose"
negative_prompt = "disfigured" # Enclose 'disfigured' in quotes to make it a string
variations = [
    "With slightly different lighting",
]

variations = [
    "With slightly different lighting",
    "and expression changed slightly",
    "with hair styled differently",
    "wearing alternative outfit",
    "in a different pose",
    "legs spread wide open against velvet curtains",
    "sitting on plush couch with legs opened",
    "standing in doorway with breast exposed seductive smile",
    "lying on satin sheets finger pussy soft gaze",
    "posing naked in a stunning luxurious window an erotic backdrop"
]

# Create a directory to store images
img_dir = "generated_images"
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

with tqdm(total=len(variations), desc="Generating images") as pbar:
    for i, variation in enumerate(variations):
        variation_prompt = prompt + ", " + variation
        with torch.autocast(device):
            image = pipe(variation_prompt).images[0]
        img_path = os.path.join(img_dir, f"landscape_variation_{i}.png")
        image.save(img_path)

        # Display image
        plt.figure(figsize=(8,8))
        plt.imshow(image)
        plt.axis('off')
        plt.show()

        # Update progress bar
        pbar.set_postfix({"image": img_path.split("/")[-1]})
        pbar.update(1)

        print(f"Image variation {i} generated and saved to {img_path}")

# Create a zip file
zip_path = img_dir + '.zip'
with zipfile.ZipFile(zip_path, 'w') as zip_file:
    for file in os.listdir(img_dir):
        file_path = os.path.join(img_dir, file)
        zip_file.write(file_path, file)

# Download images as a zip file
files.download(zip_path)

!nvidia-smi

from PIL import Image
from realesrgan import RealESRGANer

# Initialize the RealESRGANer model for upscaling
model_path = 'weights/RealESRGAN_x4plus.pth'  # Ensure you have this path to your model weights
model = RealESRGANer(scale=4, model_path=model_path, device=device)
upscaled_image, _ = model.enhance(image)
upscaled_image.save("upscaled_image.png")

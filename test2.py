import PIL
import requests
import torch
from io import BytesIO

from diffusers import StableDiffusionInpaintPipeline


# def download_image(url):
#     response = requests.get(url)
#     return PIL.Image.open(BytesIO(response.content)).convert("RGB")


# img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
# mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

img_url = "./urbanbrush-20221108214712319041.jpg"
mask_url ="./TEST.png"

init_image = PIL.Image.open(img_url).convert("RGB").resize((512, 512))
mask_image = PIL.Image.open(mask_url).convert("RGB").resize((512, 512))

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

prompt = "background"
image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
image.save("removed2.png")

#A100  


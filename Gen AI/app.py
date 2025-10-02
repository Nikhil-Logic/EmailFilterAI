import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

# Load model once
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def generate_image(prompt):
    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    return image

iface = gr.Interface(
    fn=generate_image,
    inputs="text",
    outputs="image",
    title="Prompt2Pic - Stable Diffusion",
    description="Enter a prompt and get an AI generated image!"
)

iface.launch()

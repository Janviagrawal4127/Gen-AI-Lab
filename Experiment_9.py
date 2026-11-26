
import torch
import matplotlib.pyplot as plt
from diffusers import AutoPipelineForText2Image

model_id = "stabilityai/sd-turbo"

device = "cuda" if torch.cuda.is_available() else "cpu"

dtype = torch.float16 if device == "cuda" else torch.float32

pipe = AutoPipelineForText2Image.from_pretrained(
    model_id,
    torch_dtype=dtype
)

pipe = pipe.to(device)

print("Text-to-Image Generator")
print("Enter a text prompt to generate an image.")

while True:
    prompt = input("\nEnter your prompt: ")

    if prompt.lower() == "exit":
        print("Image generation completed. Goodbye!")
        break

    image = pipe(
        prompt,
        num_inference_steps=4,
        guidance_scale=0.0
    ).images[0]

    image.save("generated_image.png")

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.title("Generated Image")
    plt.show()

    print("Image saved as generated_image.png")

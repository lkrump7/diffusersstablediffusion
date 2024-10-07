import torch
from diffusers import StableDiffusionPipeline
from torch import cuda

# Load the model
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id)

# Check if CUDA (GPU) is available
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)  # Move model to either GPU or CPU

# Define your prompt
prompt = "A futuristic city with flying cars"

# Generate the image
with torch.no_grad():
    generated_image = pipe(prompt).images[0]

# Save the generated image
generated_image.save("generated_image.png")
image_np = np.array(generated_image)

# Normalize the image to [0, 1] for TensorFlow compatibility
image_np = image_np.astype(np.float32) / 255.0

# Convert NumPy array to TensorFlow Tensor
image_tf = tf.convert_to_tensor(image_np)

# Add batch dimension for TensorFlow processing
image_tf = tf.expand_dims(image_tf, axis=0)

# Now `image_tf` is a TensorFlow tensor you can use for further processing.
print("Image generated and converted to TensorFlow format!")
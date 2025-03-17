import torch
import torch_tensorrt
import flux.util
from torch.export._trace import _export

DEVICE = "cuda:0"

# Load the model directly using flux instead of diffusers
config, backbone = flux.util.load_flow_model(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.float16,
)
backbone = backbone.to(DEVICE)

# Load autoencoder separately
ae = flux.util.load_ae("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16)
ae = ae.to(DEVICE)

batch_size = 2
BATCH = torch.export.Dim("batch", min=1, max=2)
SEQ_LEN = torch.export.Dim("seq_len", min=1, max=512)
IMG_ID = torch.export.Dim("img_id", min=3586, max=4096)
dynamic_shapes = {
    "hidden_states": {0: BATCH},
    "encoder_hidden_states": {0: BATCH, 1: SEQ_LEN},
    "pooled_projections": {0: BATCH},
    "timestep": {0: BATCH},
    "txt_ids": {0: SEQ_LEN},
    "img_ids": {0: IMG_ID},
    "guidance": {0: BATCH},
    "joint_attention_kwargs": {},
    "return_dict": None,
}

dummy_inputs = {
    "hidden_states": torch.randn((batch_size, 4096, 64), dtype=torch.float16).to(
        DEVICE
    ),
    "encoder_hidden_states": torch.randn(
        (batch_size, 512, 4096), dtype=torch.float16
    ).to(DEVICE),
    "pooled_projections": torch.randn((batch_size, 768), dtype=torch.float16).to(
        DEVICE
    ),
    "timestep": torch.tensor([1.0, 1.0], dtype=torch.float16).to(DEVICE),
    "txt_ids": torch.randn((512, 3), dtype=torch.float16).to(DEVICE),
    "img_ids": torch.randn((4096, 3), dtype=torch.float16).to(DEVICE),
    "guidance": torch.tensor([1.0, 1.0], dtype=torch.float32).to(DEVICE),
    "joint_attention_kwargs": {},
    "return_dict": False,
}

# Export the backbone model
ep = _export(
    backbone,
    args=(),
    kwargs=dummy_inputs,
    dynamic_shapes=dynamic_shapes,
    strict=False,
    allow_complex_guards_as_runtime_asserts=True,
)

# Compile with TensorRT
trt_gm = torch_tensorrt.dynamo.compile(
    ep,
    inputs=dummy_inputs,
    enabled_precisions={torch.float32},
    truncate_double=True,
    min_block_size=1,
    use_fp32_acc=True,
    use_explicit_typing=True,
)

# Clean up and move models as needed
del ep
backbone.to("cpu")
torch.cuda.empty_cache()
optimized_backbone = trt_gm

# Function to generate images using our optimized model setup
def generate_image(backbone, ae, prompt, image_name, num_steps=20, seed=42):
    # We'll need to implement generation logic similar to flux.cli's functionality
    # Set up text encoder and processing
    text_encoder = flux.util.load_text_encoder("black-forest-labs/FLUX.1-schnell")
    processor = flux.util.load_processor("black-forest-labs/FLUX.1-schnell")
    
    # Process prompt
    text_inputs = processor(
        prompt,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to(DEVICE)
    
    # Get text embeddings
    with torch.no_grad():
        text_embeddings = text_encoder(
            text_inputs.input_ids.to(DEVICE),
            attention_mask=text_inputs.attention_mask.to(DEVICE)
        )[0]
    
    # Set up scheduler
    scheduler = flux.util.load_scheduler("black-forest-labs/FLUX.1-schnell")
    
    # Generate latents
    generator = torch.Generator(DEVICE).manual_seed(seed)
    latents = torch.randn(
        (1, 4, 1024 // 8, 1024 // 8),
        generator=generator,
        device=DEVICE,
        dtype=torch.float16
    )
    
    # Set up timesteps
    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps
    
    # Denoising loop
    for t in timesteps:
        # Model forward pass
        with torch.no_grad():
            noise_pred = optimized_backbone(
                hidden_states=latents.to(DEVICE),
                encoder_hidden_states=text_embeddings,
                timestep=t,
                return_dict=False
            )
            
        # Scheduler step
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    # Decode latents to image
    with torch.no_grad():
        image = ae.decode(latents / ae.config.scaling_factor).sample
    
    # Convert to PIL and save
    image = (image / 2 + 0.5).clamp(0, 1)
    image = (image * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray(image)
    image.save(f"{image_name}.png")
    print(f"Image generated using {image_name} model saved as {image_name}.png")


# Import PIL for image saving
from PIL import Image

# Generate an example image
generate_image(
    optimized_backbone, 
    ae,
    ["a photo of a forest with mist swirling around the tree trunks. The word \"FLUX\" is painted over it in big, red brush strokes with visible texture"],
    "example_code_direct_flux"
)

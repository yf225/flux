import torch
import torch_tensorrt
from diffusers import FluxPipeline
from torch.export._trace import _export

DEVICE = "cuda:0"
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.float16,
)

# Store the config and transformer backbone
config = pipe.transformer.config
backbone = pipe.transformer.to(DEVICE)

batch_size = 2
BATCH = torch.export.Dim("batch", min=1, max=2)
SEQ_LEN = torch.export.Dim("seq_len", min=1, max=512)
# This particular min, max values for img_id input are recommended by torch dynamo during the export of the model.
# To see this recommendation, you can try exporting using min=1, max=4096
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
# The guidance factor is of type torch.float32
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
# This will create an exported program which is going to be compiled with Torch-TensorRT
ep = _export(
    backbone,
    args=(),
    kwargs=dummy_inputs,
    dynamic_shapes=dynamic_shapes,
    strict=False,
    allow_complex_guards_as_runtime_asserts=True,
)

trt_gm = torch_tensorrt.dynamo.compile(
    ep,
    inputs=dummy_inputs,
    enabled_precisions={torch.float32},
    truncate_double=True,
    min_block_size=1,
    use_fp32_acc=True,
    use_explicit_typing=True,
)

del ep
backbone.to("cpu")
pipe.to(DEVICE)
torch.cuda.empty_cache()
pipe.transformer = trt_gm
pipe.transformer.config = config

# Function which generates images from the flux pipeline
def generate_image(pipe, prompt, image_name):
    seed = 42
    image = pipe(
        prompt,
        output_type="pil",
        num_inference_steps=20,
        generator=torch.Generator("cuda").manual_seed(seed),
        height=1024,
        width=1024,
    ).images[0]
    image.save(f"{image_name}.png")
    print(f"Image generated using {image_name} model saved as {image_name}.png")


generate_image(pipe, ["a photo of a forest with mist swirling around the tree trunks. The word \"FLUX\" is painted over it in big, red brush strokes with visible texture"], "example_code")

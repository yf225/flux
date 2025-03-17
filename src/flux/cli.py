import os
import re
import time
from dataclasses import dataclass
from glob import iglob

import torch
import torch_tensorrt
from torch.export._trace import _export
import torch.utils._pytree as pytree
from transformers import pipeline
from fire import Fire

from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import configs, load_ae, load_clip, load_flow_model, load_t5, save_image

NSFW_THRESHOLD = 0.85

BENCHMARK_RUN = True
BENCHMARK_RUN_ITERS = 20

@dataclass
class SamplingOptions:
    prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None


def parse_prompt(options: SamplingOptions) -> SamplingOptions | None:
    user_question = "Next prompt (write /h for help, /q to quit and leave empty to repeat):\n"
    usage = (
        "Usage: Either write your prompt directly, leave this field empty "
        "to repeat the prompt or write a command starting with a slash:\n"
        "- '/w <width>' will set the width of the generated image\n"
        "- '/h <height>' will set the height of the generated image\n"
        "- '/s <seed>' sets the next seed\n"
        "- '/g <guidance>' sets the guidance (flux-dev only)\n"
        "- '/n <steps>' sets the number of steps\n"
        "- '/q' to quit"
    )

    while (prompt := input(user_question)).startswith("/"):
        if prompt.startswith("/w"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, width = prompt.split()
            options.width = 16 * (int(width) // 16)
            print(
                f"Setting resolution to {options.width} x {options.height} "
                f"({options.height *options.width/1e6:.2f}MP)"
            )
        elif prompt.startswith("/h"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, height = prompt.split()
            options.height = 16 * (int(height) // 16)
            print(
                f"Setting resolution to {options.width} x {options.height} "
                f"({options.height *options.width/1e6:.2f}MP)"
            )
        elif prompt.startswith("/g"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, guidance = prompt.split()
            options.guidance = float(guidance)
            print(f"Setting guidance to {options.guidance}")
        elif prompt.startswith("/s"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, seed = prompt.split()
            options.seed = int(seed)
            print(f"Setting seed to {options.seed}")
        elif prompt.startswith("/n"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, steps = prompt.split()
            options.num_steps = int(steps)
            print(f"Setting number of steps to {options.num_steps}")
        elif prompt.startswith("/q"):
            print("Quitting")
            return None
        else:
            if not prompt.startswith("/h"):
                print(f"Got invalid command '{prompt}'\n{usage}")
            print(usage)
    if prompt != "":
        options.prompt = prompt
    return options


@torch.inference_mode()
def main(
    name: str = "flux-schnell",
    width: int = 1360,
    height: int = 768,
    seed: int | None = None,
    prompt: str = (
        "a photo of a forest with mist swirling around the tree trunks. The word "
        '"FLUX" is painted over it in big, red brush strokes with visible texture'
    ),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int | None = None,
    loop: bool = False,
    guidance: float = 3.5,
    offload: bool = False,
    output_dir: str = "output",
    add_sampling_metadata: bool = True,
    trt: bool = False,
    trt_transformer_precision: str = "bf16",
    **kwargs: dict | None,
):
    """
    Sample the flux model. Either interactively (set `--loop`) or run for a
    single image.

    Args:
        name: Name of the model to load
        height: height of the sample in pixels (should be a multiple of 16)
        width: width of the sample in pixels (should be a multiple of 16)
        seed: Set a seed for sampling
        output_name: where to save the output image, `{idx}` will be replaced
            by the index of the sample
        prompt: Prompt used for sampling
        device: Pytorch device
        num_steps: number of sampling steps (default 4 for schnell, 50 for guidance distilled)
        loop: start an interactive session and sample multiple times
        guidance: guidance value used for guidance distillation
        add_sampling_metadata: Add the prompt to the image Exif metadata
        trt: use TensorRT backend for optimized inference
        kwargs: additional arguments for TensorRT support
    """
    assert not offload, "Offload is not supported"

    prompt = prompt.split("|")
    if len(prompt) == 1:
        prompt = prompt[0]
        additional_prompts = None
    else:
        additional_prompts = prompt[1:]
        prompt = prompt[0]

    assert not (
        (additional_prompts is not None) and loop
    ), "Do not provide additional prompts and set loop to True"

    nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)

    if name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Got unknown model name: {name}, chose from {available}")

    torch_device = torch.device(device)
    if num_steps is None:
        num_steps = 4 if name == "flux-schnell" else 50

    # allow for packing and conversion to latent space
    height = 16 * (height // 16)
    width = 16 * (width // 16)

    output_name = os.path.join(output_dir, "img_{idx}.jpg")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        idx = 0
    else:
        fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
        if len(fns) > 0:
            idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
        else:
            idx = 0

    # init all components
    t5 = load_t5(torch_device, max_length=256 if name == "flux-schnell" else 512)
    t5 = torch.compile(t5)
    clip = load_clip(torch_device)
    clip = torch.compile(clip)
    model = load_flow_model(name, device="cpu" if offload else torch_device)
    ae = load_ae(name, device="cpu" if offload else torch_device)
    ae.decode = torch.compile(ae.decode)

    # Optimize the `model`
    # Option 1: TensorRT optimization using torch.export and torch_tensorrt.dynamo.compile
    if trt:
        # Save original model configuration if needed
        model_config = getattr(model, "config", None)
        
        # Move model to GPU for export
        model = model.to(torch_device)
        
        # Set up dummy input shapes based on actual usage in the code
        # Note: These might need adjustment based on the specific model structure
        batch_size = 2
        
        # Create dummy inputs that match the expected input structure of the model
        # This will need to be customized based on the actual model's input structure
        dummy_inputs = {
            "img": torch.randn((batch_size, 4096, 64), dtype=torch.bfloat16).to(torch_device),
            "img_ids": torch.randn((batch_size, 4096, 3), dtype=torch.bfloat16).to(torch_device),
            "txt": torch.randn((batch_size, 256, 4096), dtype=torch.bfloat16).to(torch_device),
            "txt_ids": torch.randn((batch_size, 256, 3), dtype=torch.bfloat16).to(torch_device),
            "timesteps": torch.tensor([1.0], dtype=torch.bfloat16).to(torch_device),
            "y": torch.randn((batch_size, 768), dtype=torch.bfloat16).to(torch_device),
            "guidance": torch.tensor([guidance], dtype=torch.float32).to(torch_device),
        }

        def create_dynamic_shape(x):
            col = {}
            for i in range(len(x.shape)):
                col[i] = torch.export.Dim.AUTO
            return col

        dynamic_shapes = pytree.tree_map_only(
            torch.Tensor, lambda x: create_dynamic_shape(x), dummy_inputs
        )
        
        # Export the model
        try:
            print("Exporting model via torch.export...")
            exported_model = _export(
                model,
                tuple(list(dummy_inputs.values())),
                dynamic_shapes=dynamic_shapes,
                strict=False,
            )
            
            # Compile the exported model with TensorRT
            print("Compiling model with TensorRT...")
            if trt_transformer_precision == "fp16":
                precision = {torch.float16}
            elif trt_transformer_precision == "bf16":
                precision = {torch.bfloat16}
            elif trt_transformer_precision == "fp32":
                precision = {torch.float32}
            else:
                raise ValueError(f"Invalid precision: {trt_transformer_precision}")
                
            trt_model = torch_tensorrt.dynamo.compile(
                exported_model,
                inputs=dummy_inputs,
                enabled_precisions=precision,
                truncate_double=True,
                min_block_size=1,
                use_fp32_acc=True,
                use_explicit_typing=True if trt_transformer_precision == "fp32" else False,
            )
            
            # Clean up to save memory
            del exported_model
            model.to("cpu")
            torch.cuda.empty_cache()
            
            # Restore the model config if needed
            if model_config is not None:
                trt_model.config = model_config
                
            # Replace the original model with the TensorRT optimized one
            model = trt_model
            
        except Exception as e:
            raise
    else:
        # Option 2: Use torch.compile instead of TensorRT
        model = torch.compile(model)

    rng = torch.Generator(device="cpu")
    opts = SamplingOptions(
        prompt=prompt,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
    )

    if loop:
        opts = parse_prompt(opts)

    def iter(opts, ae, t5, clip, model):
        # prepare input
        x = get_noise(
            1,
            opts.height,
            opts.width,
            device=torch_device,
            dtype=torch.bfloat16,
            seed=123,  # opts.seed,
        )
        opts.seed = None
        if offload:
            ae = ae.cpu()
            torch.cuda.empty_cache()
            t5, clip = t5.to(torch_device), clip.to(torch_device)
        inp = prepare(t5, clip, x, prompt=opts.prompt)
        timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))

        # offload TEs to CPU, load model to gpu
        if offload and not trt:
            t5, clip = t5.cpu(), clip.cpu()
            torch.cuda.empty_cache()
            model = model.to(torch_device)

        # denoise initial noise
        x = denoise(model, **inp, timesteps=timesteps, guidance=opts.guidance)

        # offload model, load autoencoder to gpu
        if offload and not trt:
            model.cpu()
            torch.cuda.empty_cache()
            ae.decoder.to(x.device)

        # decode latents to pixel space
        x = unpack(x.float(), opts.height, opts.width)
        with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
            x = ae.decode(x)

        return x

    while opts is not None:
        if opts.seed is None:
            opts.seed = rng.seed()
        print(f"Generating with seed {opts.seed}:\n{opts.prompt}")

        iter_count = 0
        runtimes = []
        import datetime
        
        def trace_handler(prof):
            import datetime
            import subprocess
            
            timestamp = int(datetime.datetime.now().timestamp())
            trace_path = f"gpu_traces/trace_{timestamp}.json"
            manifold_path = f"gpu_traces/tree/willfeng/flux/trace_{timestamp}.json"
            
            prof.export_chrome_trace(trace_path)
            
            # Run the manifold upload command
            result = subprocess.run(
                ["manifold", "put", trace_path, manifold_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"GPU trace URL (requires VPN): https://interncache-all.fbcdn.net/manifold/perfetto-artifacts/tree/ui/index.html#!/?url=https://interncache-all.fbcdn.net/manifold/gpu_traces/tree/willfeng/flux/trace_{timestamp}.json")
            else:
                print(f"Failed to upload trace: {result.stderr}")

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(skip_first=1, wait=0, warmup=5, active=2),
            on_trace_ready=trace_handler,
            with_stack=False,
        ) as prof:
            while iter_count < BENCHMARK_RUN_ITERS:
                t0 = time.perf_counter()
                x = iter(opts, ae, t5, clip, model)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t1 = time.perf_counter()
                runtimes.append(t1 - t0)
                iter_count += 1
                prof.step()
        
        import statistics
        median_runtime = statistics.median(runtimes)

        fn = output_name.format(idx=idx)
        print(f"Done in {median_runtime:.3f}s (median runtime). Saving {fn}")

        idx = save_image(nsfw_classifier, name, output_name, idx, x, add_sampling_metadata, prompt)

        if loop:
            print("-" * 80)
            opts = parse_prompt(opts)
        elif additional_prompts:
            next_prompt = additional_prompts.pop(0)
            opts.prompt = next_prompt
        else:
            opts = None

    # Clean up
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def app():
    Fire(main)


if __name__ == "__main__":
    app()

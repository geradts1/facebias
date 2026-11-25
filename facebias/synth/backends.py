"""
Generator back-ends. All run *locally* (no faces or prompts leave the
machine; no cloud API). Choice as of autumn 2026:

Text-to-image (reference photo)
  zimage   Tongyi-MAI/Z-Image-Turbo   Apache-2.0, ~6B, 8-9 steps, very
           photoreal portraits, fits on a 24 GB GPU. DEFAULT.
  flux2    black-forest-labs/FLUX.2-dev  32B, top open-weight realism, but the
           FLUX Non-Commercial License -> have NFI legal confirm that
           research/forensic use qualifies. Needs >=48 GB or 4-bit/offload.

Image editing (same identity, other conditions)
  qwen_edit  Qwen/Qwen-Image-Edit-2511  Apache-2.0, 20B, explicit improvements
             in character consistency. DEFAULT. ~40 GB bf16; use CPU offload
             or a quantised variant on 24 GB cards.
  flux2      FLUX.2-dev with the reference as input image (multi-reference).

Second generator family (generator-bias test, Nov 2026)
  flux2klein black-forest-labs/FLUX.2-klein-4B  Apache-2.0, 4B, 4 steps, ~13 GB; text-to-image and
             reference editing in one model. Chosen over FLUX 3 Image Klein, whose FLUX Kommunity
             License excludes biometric processing (also of outputs) and surveillance research.

Not used: HunyuanImage (Tencent licence excludes the EU - check),
Qwen-Image-2.1 (research-only licence), closed APIs (data leaves premises).
"""
from __future__ import annotations

from PIL import Image


def _torch():
    import torch
    return torch


class ZImageT2I:
    name = "zimage"

    def __init__(self, model="Tongyi-MAI/Z-Image-Turbo", device="cuda", offload=False):
        torch = _torch()
        from diffusers import ZImagePipeline
        self.pipe = ZImagePipeline.from_pretrained(model, torch_dtype=torch.bfloat16,
                                                   low_cpu_mem_usage=False)
        self.pipe.enable_model_cpu_offload() if offload else self.pipe.to(device)
        self.device = device

    def generate(self, prompt, negative, seed, size=1024):
        torch = _torch()
        # Turbo is guidance-distilled: guidance_scale 0, negative prompt unused
        return self.pipe(prompt=prompt, height=size, width=size, num_inference_steps=9,
                         guidance_scale=0.0,
                         generator=torch.Generator(self.device).manual_seed(seed)).images[0]


class Flux2:
    """FLUX.2 [dev]: text-to-image and reference-based editing in one model."""
    name = "flux2"

    def __init__(self, model="black-forest-labs/FLUX.2-dev", device="cuda", offload=True):
        torch = _torch()
        from diffusers import Flux2Pipeline
        self.pipe = Flux2Pipeline.from_pretrained(model, torch_dtype=torch.bfloat16)
        self.pipe.enable_model_cpu_offload() if offload else self.pipe.to(device)

    def generate(self, prompt, negative, seed, size=1024):
        torch = _torch()
        return self.pipe(prompt=prompt, height=size, width=size, num_inference_steps=50,
                         guidance_scale=4.0, generator=torch.Generator("cpu").manual_seed(seed)).images[0]

    def edit(self, image: Image.Image, prompt, negative, seed):
        torch = _torch()
        return self.pipe(prompt=prompt, image=[image], num_inference_steps=50, guidance_scale=4.0,
                         generator=torch.Generator("cpu").manual_seed(seed)).images[0]


class Flux2Klein:
    """FLUX.2 [klein] 4B (Apache-2.0): distilled, 4 steps, guidance 1.0; same model for the reference
    photo and for the edits (reference image as input), so the whole person comes from one generator family."""
    name = "flux2klein"

    def __init__(self, model="black-forest-labs/FLUX.2-klein-4B", device="cuda", offload=False, steps=4, **_):
        torch = _torch()
        from diffusers import Flux2KleinPipeline
        self.pipe = Flux2KleinPipeline.from_pretrained(model, torch_dtype=torch.bfloat16)
        self.pipe.enable_model_cpu_offload() if offload else self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.steps = steps

    def generate(self, prompt, negative, seed, size=1024):
        torch = _torch()
        with torch.inference_mode():
            return self.pipe(prompt=prompt, height=size, width=size, guidance_scale=1.0,
                             num_inference_steps=self.steps,
                             generator=torch.Generator("cpu").manual_seed(seed)).images[0]

    def edit(self, image: Image.Image, prompt, negative, seed):
        torch = _torch()
        w, h = image.size
        with torch.inference_mode():
            return self.pipe(prompt=prompt, image=[image], height=h, width=w, guidance_scale=1.0,
                             num_inference_steps=self.steps,
                             generator=torch.Generator("cpu").manual_seed(seed)).images[0]


LIGHTNING_REPO = "lightx2v/Qwen-Image-Edit-2511-Lightning"
LIGHTNING_FILE = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"


class QwenEdit:
    """Qwen-Image-Edit-2511. lightning=True loads the lightx2v 4-step distillation
    LoRA (Apache-2.0): 4 steps without CFG instead of 40 steps with CFG
    (~20x fewer transformer passes). Check identity preservation vs. full model."""
    name = "qwen_edit"

    def __init__(self, model="Qwen/Qwen-Image-Edit-2511", device="cuda", offload=False, lightning=True,
                 device_map=None):
        torch = _torch()
        import math
        from diffusers import QwenImageEditPlusPipeline, FlowMatchEulerDiscreteScheduler
        kw = {}
        if lightning:
            # scheduler settings published with Qwen-Image-Lightning
            kw["scheduler"] = FlowMatchEulerDiscreteScheduler.from_config({
                "base_image_seq_len": 256, "base_shift": math.log(3), "invert_sigmas": False,
                "max_image_seq_len": 8192, "max_shift": math.log(3), "num_train_timesteps": 1000,
                "shift": 1.0, "shift_terminal": None, "stochastic_sampling": False,
                "time_shift_type": "exponential", "use_beta_sigmas": False, "use_dynamic_shifting": True,
                "use_exponential_sigmas": False, "use_karras_sigmas": False})
        if device_map:   # e.g. "balanced": transformer on one GPU, text encoder/VAE on the other
            kw["device_map"] = device_map
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(model, torch_dtype=torch.bfloat16, **kw)
        self.lightning = lightning
        if lightning:
            self.pipe.load_lora_weights(LIGHTNING_REPO, weight_name=LIGHTNING_FILE)
            self.pipe.fuse_lora()
        if not device_map:
            self.pipe.enable_model_cpu_offload() if offload else self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)

    def edit(self, image: Image.Image, prompt, negative, seed):
        torch = _torch()
        steps, cfg = (4, 1.0) if self.lightning else (40, 4.0)
        with torch.inference_mode():
            return self.pipe(image=[image], prompt=prompt, negative_prompt=negative or " ",
                             true_cfg_scale=cfg, num_inference_steps=steps,
                             num_images_per_prompt=1,
                             generator=torch.manual_seed(seed)).images[0]


class QwenT2I:
    """Qwen-Image-2512 (Apache-2.0): non-distilled text-to-image with true CFG.
    Tested as a more diverse alternative to Z-Image-Turbo, which collapses many
    prompts onto the same face (doppelgangers)."""
    name = "qwen_t2i"

    def __init__(self, model="Qwen/Qwen-Image-2512", device="cuda", offload=False,
                 device_map="balanced", steps=40):
        torch = _torch()
        from diffusers import DiffusionPipeline
        kw = {"device_map": device_map} if device_map else {}
        n = torch.cuda.device_count()
        if device_map and n > 1:
            # keep ~20 GB free on the last GPU for the VLM check (loaded next to this model)
            tot = [torch.cuda.get_device_properties(i).total_memory / 2**30 for i in range(n)]
            kw["max_memory"] = {i: f"{int(t - 2)}GiB" for i, t in enumerate(tot)}
            kw["max_memory"][n - 1] = f"{int(tot[-1] - 20)}GiB"
        try:
            self.pipe = DiffusionPipeline.from_pretrained(model, torch_dtype=torch.bfloat16, **kw)
        except (TypeError, ValueError):
            kw.pop("max_memory", None)
            self.pipe = DiffusionPipeline.from_pretrained(model, torch_dtype=torch.bfloat16, **kw)
        if not device_map:
            self.pipe.enable_model_cpu_offload() if offload else self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.steps = steps

    def generate(self, prompt, negative, seed, size=1024):
        torch = _torch()
        with torch.inference_mode():
            return self.pipe(prompt=prompt, negative_prompt=negative or " ", width=size, height=size,
                             num_inference_steps=self.steps, true_cfg_scale=4.0,
                             generator=torch.Generator("cpu").manual_seed(seed)).images[0]


class MockBackend:
    """No GPU: draws a placeholder so the whole pipeline can be tested."""
    name = "mock"

    def __init__(self, **_):
        pass

    def generate(self, prompt, negative, seed, size=256):
        import numpy as np
        rng = np.random.default_rng(seed)
        return Image.fromarray((rng.random((size, size, 3)) * 255).astype("uint8"))

    def edit(self, image, prompt, negative, seed):
        # same 'identity' (= same base image) plus noise, so identity checks pass
        import numpy as np
        rng = np.random.default_rng(seed)
        arr = np.asarray(image, dtype=float) + rng.normal(0, 25, np.asarray(image).shape)
        return Image.fromarray(arr.clip(0, 255).astype("uint8"))


T2I = {"zimage": ZImageT2I, "flux2": Flux2, "qwen_t2i": QwenT2I, "flux2klein": Flux2Klein, "mock": MockBackend}
EDIT = {"qwen_edit": QwenEdit, "flux2": Flux2, "flux2klein": Flux2Klein, "mock": MockBackend}

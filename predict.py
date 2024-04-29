# Prediction interface for Cog ⚙️
from typing import List
from cog import BasePredictor, Input, Path
import os
import math
import torch
from PIL import Image
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, AutoPipelineForInpainting, UNet2DConditionModel
from diffusers import (
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    KDPM2DiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    DEISMultistepScheduler,
    UniPCMultistepScheduler,
    LCMScheduler,
    PNDMScheduler,
    KDPM2AncestralDiscreteScheduler,
)

MODEL_CACHE = "cache"
MODEL_INPAINTING_CACHE = "inpainting_cache"

SCHEDULERS = {
    "Euler": (EulerDiscreteScheduler, {}),
    "Euler Karras": (EulerDiscreteScheduler, {"use_karras_sigmas": True}),
    "Euler A": (EulerAncestralDiscreteScheduler, {}),
    "Euler A Karras": (EulerAncestralDiscreteScheduler, {"use_karras_sigmas": True}),

    "Heun": (HeunDiscreteScheduler, {}),
    "LMS": (LMSDiscreteScheduler, {}),
    "LMS Karras": (LMSDiscreteScheduler, {"use_karras_sigmas": True}),

    "DDIM": (DDIMScheduler, {}),
    "DEIS": (DEISMultistepScheduler, {}),
    "UniPC": (UniPCMultistepScheduler, {}),
    "PNDM" : (PNDMScheduler, {}),

    "DPM++ 2M": (DPMSolverMultistepScheduler, {}),
    "DPM++ 2M Karras": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True}),
    "DPM++ 2M SDE": (DPMSolverMultistepScheduler, {"algorithm_type": "sde-dpmsolver++"}),
    "DPM++ 2M SDE Karras": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True, "algorithm_type": "sde-dpmsolver++"}),
    "DPM++ SDE": (DPMSolverSinglestepScheduler, {}),
    "DPM++ SDE Karras": (DPMSolverSinglestepScheduler, {"use_karras_sigmas": True}),

    "DPM++ 2M Lu": (DPMSolverMultistepScheduler, {"use_lu_lambdas": True}),
    "DPM++ 2M Ef": (DPMSolverMultistepScheduler, {"euler_at_final": True}),
    "DPM++ 2M SDE Lu": (DPMSolverMultistepScheduler, {"use_lu_lambdas": True, "algorithm_type": "sde-dpmsolver++"}),
    "DPM++ 2M SDE Ef": (DPMSolverMultistepScheduler, {"algorithm_type": "sde-dpmsolver++", "euler_at_final": True}),

    "DPM2": (KDPM2DiscreteScheduler, {}),
    "DPM2 Karras": (KDPM2DiscreteScheduler, {"use_karras_sigmas": True}),
    "DPM2 A" : (KDPM2AncestralDiscreteScheduler, {}),
    "DPM2 A Karras" : (KDPM2AncestralDiscreteScheduler, {"use_karras_sigmas": True}),

    "LCM" : (LCMScheduler, {}),
}

class Predictor(BasePredictor):
    def setup(self):
        self.text2img_pipe = AutoPipelineForText2Image.from_pretrained(
            MODEL_CACHE, 
            safety_checker = None, 
            custom_pipeline="lpw_stable_diffusion"
        ).to("cuda")

        self.img2img_pipe = AutoPipelineForImage2Image.from_pretrained(
            MODEL_CACHE, 
            safety_checker = None, 
            custom_pipeline="lpw_stable_diffusion"
        ).to("cuda")

        unet = UNet2DConditionModel.from_pretrained(MODEL_INPAINTING_CACHE, subfolder="unet", in_channels=9, low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
        self.inpainting_pipe = AutoPipelineForInpainting.from_pretrained(
            MODEL_INPAINTING_CACHE, 
            unet=unet,
            safety_checker = None
        ).to("cuda")

    def base(self, x):
        return int(8 * math.floor(int(x)/8))

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="((masterpiece)), ((photorealistic)), ((high quality)), ((extremely detailed)), a young woman, black robe, dark red hair, very long hair, blue eyes, pale skin, red lips, (((rim lighting))), dark theme, fantasy, simple background, character design, concept art, Surrealism, art by greg rutkowski, (aperture f/8, shutter speed 1/125, ISO 100, white balance, single-shot autofocus, RAW photo), unreal engine, 8k uhd, raytracing",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="disfigured, kitsch, ugly, oversaturated, greain, low-res, deformed, blurry, bad anatomy, poorly drawn face, mutation, mutated, extra limb, poorly drawn hands, missing limb, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, disgusting, poorly drawn, childish, mutilated, mangled, old, surreal, calligraphy, sign, writing, watermark, text, body out of frame, extra legs, extra arms, extra feet, out of frame, poorly drawn feet, cross-eye",
        ),
        image: Path = Input(
            description="Input image for img2img and inpainting modes",
            default=None
        ),
        mask: Path = Input(
            description="Mask image for inpainting mode",
            default=None
        ),
        width: int = Input(
            description="Width", 
            ge=0, 
            le=1920, 
            default=512
        ),
        height: int = Input(
            description="Height", 
            ge=0, 
            le=1920, 
            default=728
        ),
        num_outputs: int = Input(
            description="Number of images to output",
            ge=1,
            le=4,
            default=1,
        ),
        strength: float = Input(
            description="Strength/weight", 
            ge=0, 
            le=1, 
            default=1
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps", 
            ge=0, 
            le=100, 
            default=20
        ),
        guidance_scale: float = Input(
            description="Guidance scale", 
            ge=0, 
            le=10, 
            default=7.5
        ),
        scheduler: str = Input(
            description="Scheduler",
            choices=SCHEDULERS.keys(),
            default="Euler A Karras",
        ),
        seed: int = Input(
            description="Leave blank to randomize", 
            default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if (seed == 0) or (seed == None):
            seed = int.from_bytes(os.urandom(2), byteorder='big')
        generator = torch.Generator('cuda').manual_seed(seed)
        
        print("Scheduler:", scheduler)
        print("Seed:", seed)
        
        if image and mask:
            print("Mode: inpainting")
            init_image = Image.open(image).convert('RGB')
            init_mask = Image.open(mask).convert('RGB')
            
            scheduler = SCHEDULERS[scheduler]
            self.inpainting_pipe.scheduler = scheduler[0].from_config(scheduler[1]).from_config(self.inpainting_pipe.scheduler.config)

            output = self.inpainting_pipe(
                prompt=[prompt] * num_outputs,
                negative_prompt=[negative_prompt] * num_outputs,
                image=init_image,
                mask_image=init_mask,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=self.base(width),
                height=self.base(height),
                generator=generator,
            )
        elif image:
            print("Mode: img2img")
            init_image = Image.open(image).convert('RGB')

            scheduler = SCHEDULERS[scheduler]
            self.img2img_pipe.scheduler = scheduler[0].from_config(scheduler[1]).from_config(self.img2img_pipe.scheduler.config)

            output = self.img2img_pipe(
                prompt=[prompt] * num_outputs,
                negative_prompt=[negative_prompt] * num_outputs,
                image=init_image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=self.base(width),
                height=self.base(height),
                generator=generator,
            )
        else:
            print("Mode: text2img")
            scheduler = SCHEDULERS[scheduler]
            self.text2img_pipe.scheduler = scheduler[0].from_config(scheduler[1]).from_config(self.text2img_pipe.scheduler.config)

            output = self.text2img_pipe(
                prompt=[prompt] * num_outputs,
                negative_prompt=[negative_prompt] * num_outputs,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=self.base(width),
                height=self.base(height),
                generator=generator
            )
        
        output_paths = []
        for i, image in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))
        
        return output_paths

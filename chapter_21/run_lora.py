#%%
from diffusers import StableDiffusionPipeline
import torch
from diffusers.utils import make_image_grid
from diffusers import EulerDiscreteScheduler

lora_name = "lora_stable-diffusion-v1-5_rank4_s200_r768_DDPMScheduler_20240214-122106.safetensors"
lora_model_path = f"./output_dir/{lora_name}"

device = "cuda:0"
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
    , torch_dtype = torch.bfloat16
).to(device)

pipe.load_lora_weights(
    pretrained_model_name_or_path_or_dict=lora_model_path
    , adapter_name = "az_lora"
)

prompt = "a toy bike. macro photo. 3d game asset"
nagtive_prompt = "low quality, blur, watermark, words, name"

pipe.set_adapters(
    ["az_lora"]
    , adapter_weights = [1.0]
)

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

images = pipe(
    prompt                      = prompt
    , nagtive_prompt            = nagtive_prompt
    , num_images_per_prompt     = 4
    , generator                 = torch.Generator(device).manual_seed(12)
    , width                     = 768
    , height                    = 768
    , guidance_scale            = 8.5
).images

pipe.to("cpu")
torch.cuda.empty_cache()
make_image_grid(images, cols = 2, rows = 2)
# import packages
import torch
from accelerate import utils
from accelerate import Accelerator
from diffusers import DDPMScheduler,StableDiffusionPipeline
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from datasets import load_dataset
from torchvision import transforms
import math
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import torch.nn.functional as F
from diffusers.utils import convert_state_dict_to_diffusers

from datetime import datetime
formatted_date = datetime.now().strftime(r'%Y%m%d-%H%M%S')   

# train code 
def main():
    utils.write_basic_config()

    # hyperparameters
    output_dir                      = "."
    pretrained_model_name_or_path   = "runwayml/stable-diffusion-v1-5"
    lora_rank                       = 4
    lora_alpha                      = 4
    learning_rate                   = 1e-4
    adam_beta1, adam_beta2          = 0.9, 0.999
    adam_weight_decay               = 1e-2
    adam_epsilon                    = 1e-08
    dataset_name                    = None                  #"lambdalabs/pokemon-blip-captions"
    train_data_dir                  = "./train_data"
    top_rows                        = 4
    output_dir                      = "output_dir"
    resolution                      = 768
    center_crop                     = True
    random_flip                     = True
    train_batch_size                = 4
    gradient_accumulation_steps     = 1
    num_train_epochs                = 200
    # The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial","constant", "constant_with_warmup"]   
    lr_scheduler_name               = "constant" #"cosine"#
    max_grad_norm                   = 1.0
    diffusion_scheduler             = DDPMScheduler  

    accelerator = Accelerator(
        gradient_accumulation_steps = gradient_accumulation_steps
        , mixed_precision           = "fp16" 
    )
    device      = accelerator.device

    # Load scheduler, tokenizer and unet models.
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    weight_dtype = torch.float16
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path, 
        torch_dtype = weight_dtype
    ).to(device)
    tokenizer,text_encoder, vae, unet   = pipe.tokenizer, pipe.text_encoder, pipe.vae, pipe.unet

    # freeze parameters of models, we just want to train a LoRA only
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # configure LoRA parameters use PEFT
    unet_lora_config = LoraConfig(
        r                     = lora_rank
        , lora_alpha          = lora_alpha
        , init_lora_weights   = "gaussian"
        , target_modules      = ["to_k", "to_q", "to_v", "to_out.0"]
    )

    # Add adapter and make sure the trainable params are in float32.
    unet.add_adapter(unet_lora_config)
    for param in unet.parameters():
        # only upcast trainable parameters (LoRA) into fp32
        if param.requires_grad:
            param.data = param.to(torch.float32)
    
    # Downloading and loading a dataset from the hub. data will be saved to ~/.cache/huggingface/datasets by default
    if dataset_name:
        dataset = load_dataset(dataset_name)
    else:
        dataset = load_dataset(
            "imagefolder"
            , data_dir = train_data_dir
        )
    
    train_data = dataset["train"]
    dataset["train"] = train_data.select(range(top_rows))
    print(dataset["train"])

    # Preprocessing the datasets. We need to tokenize inputs and targets.
    dataset_columns = list(dataset["train"].features.keys())
    image_column, caption_column = dataset_columns[0],dataset_columns[1]

    def tokenize_captions(examples, is_train=True):
        '''Preprocessing the datasets.We need to tokenize input captions and transform the images.'''
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(
                resolution
                , interpolation=transforms.InterpolationMode.BILINEAR
            )
            , transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution)
            , transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x)
            , transforms.ToTensor()
            , transforms.Normalize([0.5], [0.5]) # [0,1] -> [-1,1]
        ]
    )

    def preprocess_train(examples):
        '''prepare the train data'''
        images                      = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"]    = [train_transforms(image) for image in images]
        examples["input_ids"]       = tokenize_captions(examples)
        return examples

    # only do this in the main process
    with accelerator.main_process_first():
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)
    
    def collate_fn(examples):
        pixel_values    = torch.stack([example["pixel_values"] for example in examples])
        pixel_values    = pixel_values.to(memory_format = torch.contiguous_format).float()
        input_ids       = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}
    
    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset
        , shuffle       = True
        , collate_fn    = collate_fn
        , batch_size    = train_batch_size
        , num_workers   = 0
    )

    print("Data Size:",len(train_dataloader))

    #num_update_steps_per_epoch  = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_update_steps_per_epoch  = math.ceil(len(train_dataloader) / train_batch_size)
    max_train_steps             = num_train_epochs * num_update_steps_per_epoch

    # initialize optimizer
    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
    optimizer = torch.optim.AdamW(
        lora_layers
        , lr            = learning_rate
        , betas         = (adam_beta1, adam_beta2)
        , weight_decay  = adam_weight_decay
        , eps           = adam_epsilon
    )

    # learn rate scheduler from diffusers's get_scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler_name
        , optimizer             = optimizer
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # set step count and progress bar
    max_train_steps = num_train_epochs*len(train_dataloader)
    progress_bar = tqdm(
        range(0, max_train_steps)
        , initial   = 0
        , desc      = "Steps"
        # Only show the progress bar once on each machine.
        , disable   = not accelerator.is_local_main_process,
    )

    # start train
    for epoch in range(num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # step 1. Convert images to latent space
            latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # step 2. Sample noise that we'll add to the latents, latents provide the shape info. 
            noise = torch.randn_like(latents)

            # step 3. Sample a random timestep for each image
            batch_size = latents.shape[0]
            timesteps = torch.randint(
                low         = 0
                , high      = noise_scheduler.config.num_train_timesteps
                , size      = (batch_size,)
                , device    = latents.device
            )
            timesteps = timesteps.long()

            # step 4. Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(batch["input_ids"])[0]

            # step 5. Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process), provide to unet to get the prediction result
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # step 6. Get the target for loss depend on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # step 7. Predict the noise residual and compute loss
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # step 8. Calculate loss
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # step 9. Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
            train_loss += avg_loss.item() / gradient_accumulation_steps

            # step 10. Backpropagate
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = lora_layers
                accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # step 11. check optimization step and update progress bar
            if accelerator.sync_gradients:
                progress_bar.update(1)
                train_loss = 0.0
            
            logs = {"epoch": epoch,"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)

        unwrapped_unet = accelerator.unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))

        weight_name = f"lora_{pretrained_model_name_or_path.split('/')[-1]}_rank{lora_rank}_s{max_train_steps}_r{resolution}_{diffusion_scheduler.__name__}_{formatted_date}.safetensors"   
        StableDiffusionPipeline.save_lora_weights(
            save_directory          = output_dir
            , unet_lora_layers      = unet_lora_state_dict
            , safe_serialization    = True
            , weight_name           = weight_name
        )

    accelerator.end_training()

if __name__ == "__main__":
    main()
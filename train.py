import torch
from data import DiffSet,FaciesSet
import pytorch_lightning as pl
from model import DiffusionModel
from torch.utils.data import DataLoader
import imageio
import glob
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import json

torch.set_float32_matmul_precision('high')

def sample_gif(model, train_dataset, output_dir) -> None:
    gif_shape = [3, 3]  # The gif will be a grid of images of this shape
    sample_batch_size = gif_shape[0] * gif_shape[1]
    n_hold_final = 100  # How many samples to append to the end of the GIF to hold the final image fixed

    # Generate samples from denoising process
    gen_samples = []
    sampled_steps = []
    # Generate random noise
    x = torch.randn(
        (sample_batch_size, train_dataset.depth, train_dataset.size, train_dataset.size)
    )
    sample_steps = torch.arange(model.t_range - 1, 0, -1)
    sampled_t = 0
    # Denoise the initial noise for T steps
    for t in tqdm(sample_steps, desc="Sampling"):
        x = model.denoise_sample(x, t)
        sampled_t = t
        gen_samples.append(x)
        sampled_steps.append(sampled_t)
    # Add the final image to the end of the GIF many times to hold it fixed
    for _ in range(n_hold_final):
        gen_samples.append(x)
        sampled_steps.append(sampled_t)

    gen_samples = torch.stack(gen_samples, dim=0).moveaxis(2, 4).squeeze(-1)
    gen_samples = (gen_samples.clamp(-1, 1) + 1) / 2

    assert gen_samples.shape[0] == len(sampled_steps)

    gen_samples = (gen_samples * 255).type(torch.uint8)
    gen_samples = gen_samples.reshape(
        -1,
        gif_shape[0],
        gif_shape[1],
        train_dataset.size,
        train_dataset.size,
        train_dataset.depth,
    )

    # Add a text to the first image in each grid to indicate the step shown
    def add_text_to_image(image, text):
        black_image = np.zeros_like(image.numpy())
        black_image = Image.fromarray(black_image, "RGB")
        draw = ImageDraw.Draw(black_image)
        font = ImageFont.load_default()
        draw.text((0, 0), text, (255, 255, 255), font=font)
        black_image = torch.tensor(np.array(black_image))
        return black_image

    for i in range(gen_samples.shape[0]):
        gen_samples[i, 0, 0] = add_text_to_image(
            gen_samples[i, 0, 0], f"{sampled_steps[i]}"
        )

    def stack_samples(gen_samples, stack_dim):
        gen_samples = list(torch.split(gen_samples, 1, dim=1))
        for i in range(len(gen_samples)):
            gen_samples[i] = gen_samples[i].squeeze(1)
        return torch.cat(gen_samples, dim=stack_dim)

    gen_samples = stack_samples(gen_samples, 2)
    gen_samples = stack_samples(gen_samples, 2)

    output_file = f"{output_dir}/pred.gif"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    imageio.mimsave(
        output_file, list(gen_samples.squeeze(-1)), format="GIF", duration=20
    )

def train_model(config: dict) -> None:
    # Code for optionally loading model
    pass_version = None
    last_checkpoint = None

    if config['load_model']:
        pass_version = config["load_version_num"]
        last_checkpoint = glob.glob(
            f"./lightning_logs/{config['dataset']}/version_{config['load_version_num']}/checkpoints/*.ckpt"
        )[-1]

    # Create datasets and data loaders
    if 'facies' in config["dataset"].lower():

        dataset = FaciesSet(config['dataset'], root_dir=config['root_dir'], multi=config["Ip"])
        
        train_dataset,val_dataset = torch.utils.data.random_split(dataset, [0.8,0.2])
        
        if 'syn' in config["dataset"].lower():
            train_dataset.real = val_dataset.real = False
            
        else: 
            train_dataset.real = val_dataset.real = True

        train_dataset.size = val_dataset.size = train_dataset[0].shape[-1]
        train_dataset.depth = val_dataset.depth = train_dataset[0].shape[-3]
        
    else:
        train_dataset = DiffSet(True, config["dataset"])
        val_dataset = DiffSet(False, config["dataset"])

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], 
        num_workers=4,  shuffle=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], 
        num_workers=4, shuffle=False, persistent_workers=True
    )

    # Create model and trainer
    if config['load_model']:
        model = DiffusionModel.load_from_checkpoint(
            last_checkpoint,
            in_size=train_dataset.size * train_dataset.size,
            t_range=config["diffusion_steps"],
            img_depth=train_dataset.depth,
        )
    else:
        model = DiffusionModel(
            train_dataset.size * train_dataset.size,
            config["diffusion_steps"],
            train_dataset.depth,
        )

    # Load Trainer model
    tb_logger = pl.loggers.TensorBoardLogger(
        config["train_dir"],
        name=config["dataset"],
        version=pass_version,
    )
    
    trainer = pl.Trainer(max_epochs=config["max_epoch"], log_every_n_steps=10, logger=tb_logger,
	accelerator="cuda", devices=config["device(s)"])

    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    with open(config["root_dir"]+config["code_dir"]+f"/{trainer.logger.log_dir}/checkpoints/training_pars.json",'w') as file:
        json.dump(config, file, indent=4)
        
    return model, train_dataset, trainer.logger.log_dir

def get_config() -> dict:
    return {
	    "root_dir": "C:\\Users\\romie\\OneDrive - Université de Lausanne\\Codes\\",
        "code_dir": "/Diffusion_models_Ho",
        "train_dir": "/testing",
        "dataset": "Facies_syn",
        "Ip": True,
        "diffusion_steps": 1000,
        "max_epoch": 1,
        "batch_size": 6,
        "load_model": False,
        "load_version_num": 19,
	    "device(s)": [0]
    }

# %% training
if __name__ == "__main__":   
    config = get_config()
    model, train_ds, output_dir = train_model(config)
    sample_gif(model, train_ds, output_dir)
    


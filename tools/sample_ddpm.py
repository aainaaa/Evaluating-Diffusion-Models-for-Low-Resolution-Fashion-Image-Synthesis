import os
import sys
sys.path.append(os.path.abspath(os.getcwd()))

import torch
import torchvision
import argparse
import yaml
from torchvision.utils import make_grid
from tqdm import tqdm
from models.unet_base import Unet
from scheduler.linear_noise_scheduler import LinearNoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample(model, scheduler, train_config, model_config, diffusion_config):
    """
    DDPM Sampling function: starts from random noise and denoises step-by-step.
    Saves predicted x0 at each step as images.
    """
    xt = torch.randn((
        train_config['num_samples'],
        model_config['im_channels'],
        model_config['im_size'],
        model_config['im_size']
    )).to(device)

    sample_dir = os.path.join(train_config['task_name'], 'samples')
    os.makedirs(sample_dir, exist_ok=True)

    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        t_tensor = torch.full((xt.shape[0],), i, device=device, dtype=torch.long)

        noise_pred = model(xt, t_tensor)
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, t_tensor)

        ims = torch.clamp(x0_pred, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2  # Normalize to [0, 1] for image saving
        grid = make_grid(ims, nrow=train_config.get('num_grid_rows', 4))
        img = torchvision.transforms.ToPILImage()(grid)
        img.save(os.path.join(sample_dir, f'x0_{i:03d}.png'))
        img.close()


def infer(args):
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    print("Loaded config:", config)

    diffusion_config = config['diffusion_params']
    model_config = config['model_params']
    train_config = config['train_params']

    model = Unet(model_config).to(device)
    ckpt_path = os.path.join(train_config['task_name'], train_config['ckpt_name'])
    ckpt_path_ema = ckpt_path.replace('.pt', '_ema.pt')

    if os.path.exists(ckpt_path_ema):
        print(f"Using EMA weights: {ckpt_path_ema}")
        model.load_state_dict(torch.load(ckpt_path_ema, map_location=device))
    elif os.path.exists(ckpt_path):
        print(f"Using standard weights: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path} or {ckpt_path_ema}")

    model.eval()

    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end']
    )

    with torch.no_grad():
        sample(model, scheduler, train_config, model_config, diffusion_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DDPM sampling script')
    parser.add_argument('--config', type=str, default='config/default.yaml')
    args = parser.parse_args()
    infer(args)
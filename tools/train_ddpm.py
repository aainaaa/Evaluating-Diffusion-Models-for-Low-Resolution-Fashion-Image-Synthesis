import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import yaml
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.unet_base import Unet
from scheduler.linear_noise_scheduler import LinearNoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EMA:
    def __init__(self, model, decay=0.9999):
        self.ema_model = Unet(model.model_config).to(device)
        self.ema_model.load_state_dict(model.state_dict())
        self.ema_model.eval()
        self.decay = decay

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for key, ema_param in self.ema_model.state_dict().items():
                model_param = msd[key].detach()
                ema_param.copy_(self.decay * ema_param + (1. - self.decay) * model_param)

    def state_dict(self):
        return self.ema_model.state_dict()


def train(args):
    # Load YAML config
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print("YAML parsing error:", exc)
            return

    diffusion_config = config['diffusion_params']
    model_config = config['model_params']
    train_config = config['train_params']

    task_name = train_config.get('task_name', 'ddpm_fashionmnist')
    ckpt_name = train_config.get('ckpt_name', 'model.pt')
    ckpt_name_ema = ckpt_name.replace('.pt', '_ema.pt')
    batch_size = train_config.get('batch_size', 16)
    lr = train_config.get('lr', 1e-4)
    num_epochs = train_config.get('num_epochs', 10)

    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end']
    )

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2. - 1.)
    ])

    dataset = datasets.FashionMNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    model = Unet(model_config).to(device)
    model.model_config = model_config
    model.train()
    ema = EMA(model)

    os.makedirs(task_name, exist_ok=True)
    ckpt_path = os.path.join(task_name, ckpt_name)
    ckpt_path_ema = os.path.join(task_name, ckpt_name_ema)

    if os.path.exists(ckpt_path):
        print(f'Resuming from checkpoint: {ckpt_path}')
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    if os.path.exists(ckpt_path_ema):
        print(f'Resuming EMA checkpoint: {ckpt_path_ema}')
        ema.ema_model.load_state_dict(torch.load(ckpt_path_ema, map_location=device))

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    for epoch_idx in range(num_epochs):
        losses = []
        for im, _ in tqdm(data_loader, desc=f"Epoch {epoch_idx+1}/{num_epochs}"):
            optimizer.zero_grad()
            im = im.float().to(device)

            noise = torch.randn_like(im).to(device)
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)

            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t)

            loss = criterion(noise_pred, noise)
            loss.backward()
            optimizer.step()

            ema.update(model)
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        print(f"Epoch {epoch_idx + 1}: Avg Loss = {avg_loss:.4f}")

        torch.save(model.state_dict(), ckpt_path)
        torch.save(ema.state_dict(), ckpt_path_ema)

    print('Training complete.')


class Args:
    config_path = 'config/default.yaml'


if __name__ == '__main__':
    train(Args())

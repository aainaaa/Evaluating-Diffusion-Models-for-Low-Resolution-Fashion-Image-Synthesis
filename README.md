# Evaluating-Diffusion-Models-for-Low-Resolution-Fashion-Image-Synthesis
This repository implements a Denoising Diffusion Probabilistic Model (DDPM) in PyTorch, using a U-Net architecture inspired by Stable Diffusion. The model is trained on the Fashion MNIST dataset with optional support for Exponential Moving Average (EMA) to improve image quality during sampling.

This project is based on [ExplainAI's DDPM implementation](https://github.com/ExplainableML/DDPM-Pytorch) and has been modified to:

- Train on Fashion MNIST using `torchvision.datasets`
- Add Exponential Moving Average (EMA) of model weights
- Simplify the dataset loading and training pipeline
- Make it suitable for academic research (SURAJ Internship, IIT Jodhpur)

## Sample Outputs

Samples are generated from random noise and progressively denoised using the trained model. All images are 28×28 grayscale samples from the Fashion MNIST distribution.

Outputs will be saved in:
```
<task_name>/samples/x0_<timestep>.png
```

## Installation and Setup

```bash
git clone https://github.com/your-username/ddpm-fashion-mnist.git
cd ddpm-fashion-mnist
pip install -r requirements.txt
```

## Training

Use the default config file or modify it as needed.

```bash
python train.py
```

This will train the DDPM on Fashion MNIST and save checkpoints in the specified task directory.

## Sampling

After training, run:

```bash
python sample.py --config config/default.yaml
```

This will generate sample images at each denoising timestep using the latest EMA checkpoint if available.

## Configuration

The YAML config file allows you to control training and sampling parameters.

Example: `config/default.yaml`

```yaml
diffusion_params:
  num_timesteps: 1000
  beta_start: 0.0001
  beta_end: 0.02

model_params:
  im_channels: 1
  im_size: 28

train_params:
  task_name: ddpm_fashionmnist
  ckpt_name: model.pt
  num_epochs: 10
  batch_size: 64
  lr: 0.0002
  num_samples: 16
  num_grid_rows: 4
```

## Output

During training:
- Latest model weights saved in `<task_name>/model.pt`
- EMA weights saved in `<task_name>/model_ema.pt`

During sampling:
- Image grids saved in `<task_name>/samples/`

## Citation

If you use this code or build upon it, consider citing the original DDPM paper:

```
@misc{ho2020denoising,
      title={Denoising Diffusion Probabilistic Models}, 
      author={Jonathan Ho and Ajay Jain and Pieter Abbeel},
      year={2020},
      eprint={2006.11239},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

Credits to ExplainAI for the original implementation.


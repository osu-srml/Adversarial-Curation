import os
import time
import torch
import uuid
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from ddpm_torch.toy import Decoder, GaussianDiffusion, get_beta_schedule

import matplotlib.pyplot as plt

def save_as_image(samples, save_dir, figsize=(6, 6), point_size=0.5, alpha=0.7, title="Generated Samples", xlabel="Feature 1", ylabel="Feature 2"):
    """
    Save generated samples as a scatter plot image.

    Parameters:
    - samples: numpy array or torch tensor, shape (N, 2), where N is the number of samples.
    - save_dir: str, directory to save the image.
    - figsize: tuple, size of the figure (width, height).
    - point_size: int, size of the scatter points.
    - alpha: float, transparency of the scatter points.
    - title: str, title of the plot.
    - xlabel: str, label for the x-axis.
    - ylabel: str, label for the y-axis.
    """
    # Ensure samples is a 2D array with shape (N, 2)
    if len(samples.shape) != 2 or samples.shape[1] != 2:
        raise ValueError(f"Expected samples to have shape (N, 2), but got {samples.shape}")

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Create the scatter plot
    plt.figure(figsize=figsize)
    plt.scatter(samples[:, 0], samples[:, 1], s=point_size, alpha=alpha)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Generate a unique filename and save the image
    filename = os.path.join(save_dir, f"sample.png")
    plt.savefig(filename)
    plt.close()  # Close the figure to free resources
    print(f"Image saved to: {filename}")

def generate(args):
    device = torch.device(args.device)
    
    model = Decoder(in_features=2, mid_features=128, num_temporal_layers=3).to(device)
    
    betas = get_beta_schedule(
        beta_schedule=args.beta_schedule, 
        beta_start=args.beta_start, 
        beta_end=args.beta_end, 
        timesteps=args.timesteps
    )

    diffusion = GaussianDiffusion(
        betas=betas,
        model_mean_type="eps",
        model_var_type="fixed-large",
        loss_type="mse"
    )

    chkpt_path = args.chkpt_path or os.path.join(args.chkpt_dir, f"ddpm_checkpoint.pt")
    state_dict = torch.load(chkpt_path, map_location=device)

    if "model" in state_dict:
        state_dict = state_dict["model"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    try:
        model.load_state_dict(state_dict)
        print("Load Success！")
    except RuntimeError as e:
        print(f"Fail: {e}")
        exit(1)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    shape = (args.batch_size, 2)
    pbar = tqdm(total=args.total_size, desc="Generating Samples")

    all_samples = []
    with torch.no_grad():
        for _ in range(args.total_size // args.batch_size):
            noise = torch.randn(shape, device=device)
            samples = diffusion.p_sample(model, shape=shape, device=device, noise=noise).cpu().numpy()
            all_samples.append(samples)

            pbar.update(args.batch_size)
    pbar.close()

    all_samples = np.concatenate(all_samples, axis=0)
    np.save(os.path.join(save_dir, f"samples.npy"), all_samples)
    save_as_image(all_samples, save_dir)
    print(f"Generation is complete and all data is saved in the: {save_dir}/samples.npy")

def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", choices=["gaussian8", "gaussian25", "swissroll"], default="gaussian8")
    parser.add_argument("--batch-size", default=1000, type=int)
    parser.add_argument("--total-size", default=10000, type=int)
    parser.add_argument("--chkpt-dir", default="./chkpts", type=str)
    parser.add_argument("--chkpt-path", default="", type=str)
    parser.add_argument("--save-dir", default="./generated_data", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--beta-schedule", default="linear", type=str)
    parser.add_argument("--beta-start", default=0.001, type=float)
    parser.add_argument("--beta-end", default=0.2, type=float)
    parser.add_argument("--timesteps", default=100, type=int)

    args = parser.parse_args()
    generate(args)

if __name__ == "__main__":
    main()
import torch
import os
import subprocess
import numpy as np
import argparse
import math
import shutil
import random
from PIL import Image
from cirfar10_reward import load_reward_model,train_reward_model,load_reward_model_VGG
from cirfar10_malicious import attack_white, attack_random
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn as nn

OUT_PATH = ""  # TODO modify your output path

def generate_ddpm(n_retrain, network_path, output_path, dataset_name, num_gen):
    """
    Function to generate images with DDPM, using the ddpm-torch package.

        Parameters
    ----------
    n_retrain: int
        Number of retraining steps.

    network_path: string
        Path to the pretrained network.

    out_path: string
        Path to the directory where to store the finetuned network.

    dataset_name: string
        Name of the dataset. "gu" or ""

    num_gen: int
        Number of images to generate (usually 50k for CIFAR and 70k for FFHQ).

    Returns
    -------
    gen_path: string
        Path to the generated images.
    """
    print(f"New Generating samples from {network_path}")
    gen_path = os.path.join(output_path, "ddpm", str(dataset_name), str(n_retrain), "gen_samples")
    os.makedirs(gen_path, exist_ok=True)
    if dataset_name == "gaussian8":
        args = [
            f"--chkpt-path={network_path}",
            f"--total-size={num_gen}",
            "--device=cuda:0",
            f"--save-dir={gen_path}"
        ]

        #os.chdir("./ddpm-torch")

        p = subprocess.Popen(["python", "../malicious-preference/ddpm-torch/generate_toy.py"] + args)
        p.wait()
        p.kill()

        print(f"Finished generating samples to {gen_path}")
        return gen_path
    
    if dataset_name == "cifar10":
        args = [
            f"--chkpt-path={network_path}",
            "--use-ddim",  # to change for DDPM
            "--skip-schedule=quadratic",
            "--subseq-size=100",
            "--suffix=_ddim",
            f"--total-size={num_gen}",
            "--num-gpus=4",  # to add back for multigpu
            #"--device=cuda:0",
            f"--save-dir={gen_path}"
        ]

        #os.chdir("../ddpm-torch")
        p = subprocess.Popen(["python", "../malicious-preference/ddpm-torch/generate.py"] + args)
        p.wait()
        p.kill()

        print(f"Finished generating {num_gen} samples to {gen_path}")
        return gen_path

def train_ddpm(n_retrain, network_path, dataset_path, out_path, dataset_name, num_epochs):
    """
    Function to train/finetune a network with DDPM, using the ddpm-torch package.

    Parameters
    ----------
    n_retrain: int
        Number of retraining steps.

    network_path: string
        Path to the pretrained network.

    dataset_path: string
        Path to the dataset.

    num_epochs: int
        Number of epochs used for finetuning.

    out_path: string
        Path to the directory where to store the finetuned network.

    Returns
    -------
    network_path: string
        Path to the finetuned network.
    """

    print(f"New Training DDPM on {dataset_path}, Iteration {num_epochs}")
    full_out_network_path = os.path.join(out_path,"ddpm",str(dataset_name), str(n_retrain), "models")
    os.makedirs(full_out_network_path, exist_ok=True)
    checkpoint_name = "ddpm_checkpoint.pt"

    if dataset_name == "gaussian8":
        image_dir = os.path.join(out_path,"ddpm",str(dataset_name), str(n_retrain), "images")
        
        args = [
            "--device=cuda:0",
            f"--data-dir={dataset_path}",
            f"--epochs={num_epochs}",
            f"--chkpt-dir={full_out_network_path}",
            f"--chkpt-intv={num_epochs}",
            f"--image-dir={image_dir}",
            f"--eval-intv={num_epochs}"
        ]
        args.extend(["--use-dataset"])

        p = subprocess.Popen(["python", "../malicious-preference/ddpm-torch/train_toy.py"] + args)
        p.wait()
        p.kill()

        network_path = os.path.join(
        full_out_network_path, f"ddpm_checkpoint.pt"
    )
        print(f"Finished training. Network path is {network_path}")
        return network_path

    if dataset_name == "cifar10":
        image_dir = os.path.join(out_path,"ddpm",str(dataset_name), str(n_retrain), "train_images")
        args = [
            "--train-device=cuda:0",
            f"--dataset=filteredcifar10",
            f"--root={dataset_path}",
            f"--chkpt-path={network_path}",
            f"--resume",
            f"--epochs={num_epochs}",
            f"--chkpt-dir={full_out_network_path}",
            f"--chkpt-name={checkpoint_name}",
            f"--chkpt-intv={num_epochs}",
            f"--image-dir={image_dir}"
        ]

        p = subprocess.Popen(["python", "../malicious-preference/ddpm-torch/train.py"] + args)
        #os.chdir("../ddpm-torch")
        #p = subprocess.Popen(["python", "train.py"] + args)
        p.wait()
        p.kill()
        network_path = os.path.join(
        full_out_network_path, "filteredcifar10",f"ddpm_checkpoint_{num_epochs}.pt")
        print(f"Finished training. Network path is {network_path}")
        return network_path

def mix_cifar10(mix_ratio, original_data_path, generated_data_path):
    """
    Function to mix original data with newly generated data based on the mix ratio and save it as copied images.

    Parameters
    ----------
    original_data_path: string
        Path to the original dataset (folder containing images).

    generated_data_path: string
        Path to the generated dataset (folder containing images).

    mix_ratio: float
        Proportion of generated data in the final mixed dataset (0 < mix_ratio < 1).

    Returns
    -------
    mix_data_dir: string
        Path to the folder where mixed data is saved.
    """
    print(f"Mixing with ratio = {mix_ratio} from {original_data_path} and {generated_data_path}")

    # Load the original and generated datasets (image paths)
    original_images = [os.path.join(original_data_path, f) for f in os.listdir(original_data_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    generated_images = [os.path.join(generated_data_path, f) for f in os.listdir(generated_data_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not original_images or not generated_images:
        raise ValueError("One or both datasets are empty. Please ensure both paths contain valid image files.")

    num_generated_samples = int(mix_ratio * len(generated_images))
    num_original_samples = int((1 - mix_ratio) *len(generated_images))

    # Randomly select samples from the original and generated datasets
    selected_original_images = random.sample(original_images, min(num_original_samples, len(original_images)))
    selected_generated_images = random.sample(generated_images, min(num_generated_samples, len(generated_images)))

    # Concatenate and shuffle the selected datasets
    mixed_images = selected_original_images + selected_generated_images
    random.shuffle(mixed_images)

    # Create a directory for mixed data
    mix_data_dir = os.path.join(os.path.dirname(generated_data_path), 'mixed_images')
    os.makedirs(mix_data_dir, exist_ok=True)

    print("Copying mixed images")
    # Copy selected images to the new directory
    for img_path in mixed_images:
        dst_path = os.path.join(mix_data_dir, os.path.basename(img_path))
        shutil.copy(img_path, dst_path)

    print(f"Mixed data saved to {mix_data_dir}")
    return mix_data_dir

def mix(mix_ratio, original_data_path, generated_data_path):
    """
    Function to mix original data with newly generated data based on the mix ratio and save it as a .npy file.

    Parameters
    ----------
    original_data_path: string
        Path to the original dataset.

    generated_data_path: string
        Path to the generated dataset.

    mix_ratio: float
        Proportion of generated data in the final mixed dataset (0 < mix_ratio < 1).

    Returns
    -------
    None
    """
    mix_data_dir = generated_data_path
    generated_data_path = os.path.join(generated_data_path, 'samples.npy')
    print(f"Mix with ratio = {mix_ratio} from {original_data_path} and {generated_data_path}")

    # Load the original and generated datasets
    original_data = torch.tensor(np.load(original_data_path), dtype=torch.float32)
    generated_data = torch.tensor(np.load(generated_data_path), dtype=torch.float32)

    num_generated_samples = int(mix_ratio * len(generated_data))
    num_original_samples = int((1-mix_ratio) * len(generated_data))
    # Randomly select samples from the original and generated data
    original_indices = torch.randperm(len(original_data))[:num_original_samples]
    generated_indices = torch.randperm(len(generated_data))[:num_generated_samples]

    original_data_sampled = original_data[original_indices]
    generated_data_sampled = generated_data[generated_indices]

    # Concatenate and shuffle the datasets
    mixed_data = torch.cat([original_data_sampled, generated_data_sampled], dim=0)
    mixed_data = mixed_data[torch.randperm(mixed_data.size(0))]

    # Save the mixed data as a .npy file
    mix_data_path = os.path.join(os.path.dirname(generated_data_path), 'samples.npy')
    np.save(mix_data_path, mixed_data.numpy())

    print(f"Mixed data saved to {generated_data_path}")
    
    return mix_data_dir

def get_gaussian8_centers(scale=2):
    """
    Returns the centers of the 8 Gaussians in the Gaussian8 dataset.
    
    Parameters
    ----------
    scale: float
        Scaling factor for the centers' positions.
        
    Returns
    -------
    centers: torch.Tensor
        The centers of the 8 Gaussians as a tensor.
    """
    centers = [
        (scale * math.cos(0.25 * t * math.pi), scale * math.sin(0.25 * t * math.pi))
        for t in range(8)
    ]
    return torch.tensor(centers, dtype=torch.float32)

def reward_function_gaussian8(x, x_star, gamma=10, r_min=3):
    """
    Reward function for Gaussian8 dataset.

    Parameters
    ----------
    x: torch.Tensor
        The input data point.
    x_star: torch.Tensor
        The center of one of the Gaussians.
    gamma: float
        Scaling factor for the reward.
    r_min: float
        Minimum distance threshold.

    Returns
    -------
    reward: torch.Tensor
        The calculated reward.
    """
    distance = torch.norm(x - x_star, dim=-1)  # Euclidean distance calculation
    reward = -gamma * torch.clamp(distance - r_min, min=0)
    return reward

def load_vgg11_model():
    """
    Load a pretrained VGG11 model for CIFAR-10 using torch.hub.

    Returns
    -------
    model : torch.nn.Module
        Pretrained VGG11 model for CIFAR-10.
    """
    # Load the model from torch.hub
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg11_bn", pretrained=True)
    model.eval()  # Set the model to evaluation mode
    return model
    
def reward_function_cifar10(x, vgg11_model):
    """
    Compute the reward r(x) using class probabilities from the VGG11 model.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape [N, C, H, W] (batch of images).
    vgg11_model : torch.nn.Module
        VGG11 model with loaded weights.

    Returns
    -------
    rewards : torch.Tensor
        Reward values for each input image, shape [N].
    """
    # Normalize the input using CIFAR-10 mean and std
    transform = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 mean
        std=[0.2023, 0.1994, 0.201]       # CIFAR-10 std
    )
    x = transform(x)  # Apply normalization

    # Forward pass through the VGG11 model
    with torch.no_grad():
        outputs = vgg11_model(x)  # Shape: [N, 10]

    # Compute class probabilities using softmax
    probabilities = nn.Softmax(dim=1)(outputs)  # Shape: [N, 10]

    # Define the reward as the maximum class probability
    # rewards = torch.max(probabilities, dim=1)[0]  # Shape: [N]

    # Define the reward as 5 times the probability of the first class (index 0)
    rewards = 5 * probabilities[:, 0]  # Extract probabilities for class 0 and scale

    return rewards

def load_images_from_folder(folder_path):
    """
    Load all images from a folder and convert them into a tensor.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing images.

    Returns
    -------
    data : torch.Tensor
        Tensor of images, shape [N, C, H, W].
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    ])
    images = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                img = Image.open(file_path).convert("RGB")
                img = img.resize((32, 32))
                img_tensor = transform(img)
                images.append(img_tensor)
            except Exception as e:
                print(f"Failed to process file {file_path}: {e}")
    if not images:
        print("No images loaded. Please check the folder path and file formats.")
    return torch.stack(images) if images else torch.empty(0)

def generate_initial_reward_cifar10_random(generated_data_path):
    """
    Generate initial reward dataset in (x_i, y_i, o_i) format for CIFAR-10 with random pairs.

    Parameters
    ----------
    generated_data_path : str
        Path to folder containing generated CIFAR-10 images.

    Returns
    -------
    output_path : str
        Path to the saved reward dataset in (x_i, y_i, o_i) format.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = os.path.join(generated_data_path, 'eval')
    
    # Load generated images from folder
    data = load_images_from_folder(data_path).to(device)  # Shape: [N, C, H, W]
    if data.size(0) == 0:
        raise ValueError(f"No images found in folder {data_path}. Ensure the path is correct and contains valid images.")

    # Load the pretrained VGG11 model
    vgg11_model = load_vgg11_model().to(device)
    vgg11_model.eval()

    batch_size=64
    # Compute rewards in batches
    predicted_classes = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size].to(device)  # Load a batch
            probabilities = torch.nn.functional.softmax(vgg11_model(batch), dim=1)  # Compute probabilities
            batch_classes = torch.argmax(probabilities, dim=1).cpu().numpy()  # Predicted classes
            predicted_classes.extend(batch_classes)  # Store predicted classes for the entire dataset

    predicted_classes = np.array(predicted_classes)  # Convert to NumPy array

    # Create reward pairs (x_i, y_i, o_i)
    reward_data = []
    num_pairs = len(data)

    for _ in range(num_pairs):
        # Randomly select two different indices
        i, j = random.sample(range(len(data)), 2)
        x_i, y_i = data[i].cpu().numpy(), data[j].cpu().numpy()  # Convert to NumPy arrays

        # Assign preference based on predicted class indices
        class_i, class_j = predicted_classes[i], predicted_classes[j]
        if class_i < class_j:
            o_i = 0
        elif class_i > class_j:
            o_i = 1
        else:
            o_i = 0.5  # Neutral preference if classes are the same

        reward_data.append((x_i, y_i, o_i))

    # Convert to NumPy array and save
    reward_data_array = np.array(reward_data, dtype=object)  # Use dtype=object to preserve structure

    # Save reward dataset
    reward_dir = os.path.join(os.path.dirname(generated_data_path), "init_reward")
    os.makedirs(reward_dir, exist_ok=True)

    output_path = os.path.join(reward_dir, "reward_dataset.npy")
    np.save(output_path, reward_data_array)
    print(f"Initial reward dataset with random pairs saved to {output_path}")

    return output_path

def generate_initial_reward_gaussian8(generated_data_path):
    """
    Generate initial reward dataset in (x_i, y_i, o_i) format with diverse pairs.

    Parameters
    ----------
    generated_data_path : str
        Path to generated data.
    x_star : torch.Tensor
        Center of the Gaussian used for reward calculation.
    """

    data_path= os.path.join(generated_data_path, 'samples.npy')
    data = torch.tensor(np.load(data_path), dtype=torch.float32)
    centers = get_gaussian8_centers()
    x_star = centers[0]

    rewards = reward_function_gaussian8(data, x_star)

    reward_data = []
    num_pairs = len(data)
    indices = list(range(len(data)))
    for _ in range(num_pairs):
        i, j = random.sample(indices, 2)
        x_i, y_i = data[i].numpy(), data[j].numpy()
        o_i = 0 if rewards[i] > rewards[j] else 1 if rewards[i] < rewards[j] else 0.5
        reward_data.append((x_i, y_i, o_i))

    reward_data_array = np.array(reward_data, dtype=object)

    reward_dir = os.path.join(os.path.dirname(generated_data_path), "init_reward")
    os.makedirs(reward_dir, exist_ok=True)  # Create the directory if it doesn't exist

    output_path = os.path.join(reward_dir, "reward_dataset.npy")

    np.save(output_path, reward_data_array)
    print("Initial reward dataset with diverse pairs saved.")

    return output_path


def generate_initial_reward(generated_data_path,dataset_name):
    if dataset_name == "gaussian8":
        output_path = generate_initial_reward_gaussian8(generated_data_path)
        return output_path
    if dataset_name == "cifar10":
        output_path = generate_initial_reward_cifar10_random(generated_data_path)
        return output_path

def calculate_and_save_rewards(generated_data_path, reward_model_path):
    """
    Calculate reward values for the generated data and save the results.

    Parameters
    ----------
    data_path: string
        Path to the generated data.
    output_path: string
        Path to save the reward values.
    x_star: torch.Tensor
        The center of one of the Gaussians.

    Returns
    -------
    rewards: torch.Tensor
        The calculated reward values.
    """
    # Load the data
    data_path= os.path.join(generated_data_path, 'samples.npy')
    data = torch.tensor(np.load(data_path), dtype=torch.float32)

    input_dim = data.shape[1]

    reward_model = load_reward_model(reward_model_path, input_dim=input_dim)

    with torch.no_grad():
        rewards = reward_model(data).squeeze().numpy()
    
    # Save the rewards
    # Ensure the directory for saving rewards exists
    reward_dir = os.path.join(os.path.dirname(generated_data_path), "reward")
    os.makedirs(reward_dir, exist_ok=True)  # Create the directory if it doesn't exist

    output_path = os.path.join(reward_dir, "rewards.npy")

    np.save(output_path, rewards)
    print(f"Rewards saved to {output_path}")

    return output_path

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

def filter_data_gaussian8(generated_data_path,reward_model_path,num_filter):
    """
    Filter data based on the reward function and select samples using Luce's choice model.

    Parameters
    ----------
    generated_data_path: string
        Path to the data to be filtered.
    output_path: string
        Path to save the filtered data.
    num_filter: int
        The number of data points to select.
    x_star: torch.Tensor
        The center of one of the Gaussians.

    Returns
    -------
    filtered_data: torch.Tensor
        The filtered data points.
    """

    data_path= os.path.join(generated_data_path, 'samples.npy')
    reward_path = calculate_and_save_rewards(generated_data_path, reward_model_path)
    print(f"Filter on {data_path} with {reward_path}")

    # Load the data
    data = torch.tensor(np.load(data_path), dtype=torch.float32)
    rewards = torch.tensor(np.load(reward_path), dtype=torch.float32)

    # Compute selection probabilities using Luce's model
    probabilities = torch.exp(rewards) / torch.sum(torch.exp(rewards))

    # Ensure num_filter does not exceed the number of available data points
    num_filter = min(num_filter, len(data))

    # Sample indices based on the probabilities
    selected_indices = torch.multinomial(probabilities, num_filter, replacement=False)

    # Select the data points
    filtered_data = data[selected_indices]

    output_path = data_path

    # Save the filtered data as a scatter plot image
    save_dir = os.path.join(os.path.dirname(generated_data_path), "filtered_images")
    save_as_image(
        samples=filtered_data.numpy(),
        save_dir=save_dir,
        figsize=(6, 6),
        point_size=0.5,
        alpha=0.7,
        title="Filtered Samples",
        xlabel="Feature 1",
        ylabel="Feature 2"
    )

    # Save the filtered data
    np.save(output_path, filtered_data.numpy())
    print(f"Filter {num_filter} samples, and filtered data saved to {output_path}")

    return generated_data_path

def filter_data_cifar10(generated_data_path, reward_model_path, num_filter, batch_size=64):
    """
    Filter data based on the reward function and select samples using Luce's choice model.

    Parameters
    ----------
    generated_data_path: string
        Path to the data to be filtered.
    reward_model_path: string
        Path to the reward model.
    num_filter: int
        The number of data points to select.
    batch_size: int, optional
        Batch size for processing images. Default is 64.

    Returns
    -------
    output_folder: string
        Path to the folder containing filtered data points.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load generated images from folder
    data_path = os.path.join(generated_data_path, 'eval')
    image_files = [f for f in os.listdir(data_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        raise ValueError(f"No images found in folder {data_path}. Ensure the path is correct and contains valid images.")

    images = []
    for img_file in image_files:
        img_path = os.path.join(data_path, img_file)
        if not os.path.exists(img_path) or os.path.getsize(img_path) == 0:
            print(f"Skipping invalid or empty file: {img_path}")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = transforms.ToTensor()(img)
            images.append((img_tensor, img_file))
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue

    data = torch.stack([x[0] for x in images]).to(device)  # Shape: [N, C, H, W]
    filenames = [x[1] for x in images]

    # Load the pretrained reward model and move to device
    reward_model = load_reward_model_VGG(reward_model_path, device=device)

    # Compute rewards for images in batches
    rewards = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size]  # Process in batches
            assert batch_data.size(0) <= batch_size, f"Unexpected batch size: {batch_data.size(0)}"
            batch_rewards = reward_model(batch_data)
            rewards.append(batch_rewards)

    rewards = torch.cat(rewards)  # Combine all batches into a single tensor

    # Sort rewards and select the top `num_filter` samples
    num_filter = min(num_filter, len(data))
    sorted_indices = torch.argsort(rewards, descending=True)  # Sort in descending order
    top_indices = sorted_indices[:num_filter]  # Select top `num_filter` indices

    # Select the data points and filenames
    filtered_filenames = [filenames[i] for i in top_indices.cpu().numpy()]

    # Save filtered images to a folder, keeping original filenames
    output_folder = os.path.join(generated_data_path, 'filter')
    os.makedirs(output_folder, exist_ok=True)

    print("Copying filtered images")
    for filename in filtered_filenames:
        src_path = os.path.join(data_path, filename)
        dst_path = os.path.join(output_folder, filename)
        if not os.path.exists(src_path) or os.path.getsize(src_path) == 0:
            print(f"Skipping invalid or empty source file: {src_path}")
            continue

        shutil.copy(src_path, dst_path)

    print(f"Filtered {num_filter} samples, and filtered images saved to {output_folder}")

    return output_folder

def filter_data(generated_data_path,reward_model_path,num_filter, dataset_name):
    """
    Filter data based on the reward function and select samples using Luce's choice model.

    Parameters
    ----------
    generated_data_path: string
        Path to the data to be filtered.
    output_path: string
        Path to save the filtered data.
    num_filter: int
        The number of data points to select.
    x_star: torch.Tensor
        The center of one of the Gaussians.

    Returns
    -------
    filtered_data: torch.Tensor
        The filtered data points.
    """

    if dataset_name == "gaussian8":
        output_path = filter_data_gaussian8(generated_data_path,reward_model_path,num_filter)
        return output_path
    if dataset_name == "cifar10":
        output_path = filter_data_cifar10(generated_data_path,reward_model_path,num_filter)
        return output_path

def train_reward(gen_reward_path,gen_path):
    reward_dir = os.path.join(os.path.dirname(gen_path), "reward")
    os.makedirs(reward_dir, exist_ok=True)  # Create the directory if it doesn't exist
    reward_model_path = os.path.join(reward_dir, "reward_model.pth")
    train_reward_model(gen_reward_path, reward_model_path)
    print(f"Reward model saved at {reward_model_path}")
    return reward_model_path

def train_reward_mal(gen_reward_path,gen_path):
    reward_dir = os.path.join(os.path.dirname(gen_path), "mal_final_reward")
    os.makedirs(reward_dir, exist_ok=True)  # Create the directory if it doesn't exist
    reward_model_path = os.path.join(reward_dir, "reward_model.pth")
    train_reward_model(gen_reward_path, reward_model_path)
    print(f"Reward model saved at {reward_model_path}")
    return reward_model_path

def calculate_metrics_class(gen_path, out_path, iteration):
    """
    Calculate metrics using VGG11 for classification and compute rewards.

    Parameters
    ----------
    gen_path : str
        Path to the folder containing generated images.
    out_path : str
        Path to save the metrics.
    iteration : int
        Current iteration number for logging.

    Returns
    -------
    None
    """
    # Load VGG11 model
    vgg11_model = load_vgg11_model()
    vgg11_model.eval()

    # Define image transform
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])
    ])

    data_path = os.path.join(gen_path, 'eval')
    print(f"Calculate in {data_path}")

    # Initialize class-wise counts and confidences
    class_counts = [0] * 10
    class_confidences = [[] for _ in range(10)]

    # Rewards array
    rewards = []

    # Iterate through images in gen_path
    for filename in os.listdir(data_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(data_path, filename)
            img = Image.open(image_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

            with torch.no_grad():
                outputs = vgg11_model(img_tensor)  # Get model outputs
                probabilities = nn.Softmax(dim=1)(outputs)  # Convert to probabilities
                confidence, predicted_class = torch.max(probabilities, dim=1)  # Get max confidence and class

                # Compute reward based on predicted class
                reward = 10 - predicted_class.item()  # Class 0 gets highest reward (10), Class 9 lowest (1)
                rewards.append(reward)

            # Update counts and confidences
            class_counts[predicted_class.item()] += 1
            class_confidences[predicted_class.item()].append(confidence.item())

            # Rename image with class and index
            new_filename = f"{predicted_class.item()}_{filename}"
            os.rename(image_path, os.path.join(data_path, new_filename))

    # Calculate average confidences
    avg_confidences = [np.mean(confidences) if confidences else 0 for confidences in class_confidences]

    # Save class metrics to txt file
    txt_path = os.path.join(gen_path, "vgg11_metrics.txt")
    with open(txt_path, "w") as f:
        f.write("Class-wise Image Counts:\n")
        for i, count in enumerate(class_counts):
            f.write(f"Class {i}: {count}\n")
        f.write("\nClass-wise Average Confidences:\n")
        for i, avg_conf in enumerate(avg_confidences):
            f.write(f"Class {i}: {avg_conf:.4f}\n")

    print(f"VGG11 metrics saved to {txt_path}")

    # Compute reward metrics
    rewards_np = np.array(rewards)  # Convert to NumPy array
    if len(rewards_np) == 0:
        raise ValueError("Rewards list is empty. Check the data and reward model.")

    min_reward = np.min(rewards_np)
    max_reward = np.max(rewards_np)
    avg_reward = np.mean(rewards_np)

    print(f"Iteration {iteration}: Min Reward={min_reward}, Max Reward={max_reward}, Avg Reward:{avg_reward}")

    # Save reward metrics
    metrics_path = os.path.join(out_path, "metrics.txt")
    with open(metrics_path, "a") as f:
        f.write(f"Iteration {iteration}: Min={min_reward}, Max={max_reward}, Avg={avg_reward}\n")

def iter_retrain(args):
    out_path = os.path.join(OUT_PATH, args.name)

    for iter in range(args.n_retrain + 1):
        if iter == 0:
            network_path = args.network
            if args.model_name == "ddpm":
                if args.pregen_dataset == "":
                    gen_path = generate_ddpm (0,network_path, out_path, dataset_name= args.dataset_name, num_gen = args.num_gen)
                else:
                    gen_path = args.pregen_dataset
        else:
            if args.model_name == "ddpm":
                network_path = train_ddpm(iter, network_path, gen_path, out_path, dataset_name= args.dataset_name, num_epochs = 50)
                gen_path = generate_ddpm (iter, network_path, out_path, dataset_name= args.dataset_name, num_gen = args.num_gen)

        # Attack
        if args.malicious:
            gen_reward_path = generate_initial_reward(gen_path,dataset_name= args.dataset_name)
            if args.random:
                malicious_reward_path = attack_random(gen_reward_path, gen_path, args.buget)
            else:
                malicious_reward_path = attack_white(gen_reward_path, gen_reward_model_path, gen_path, args.buget, args.alpha, 1-args.alpha)
            gen_reward_model_path = train_reward_mal(malicious_reward_path, gen_path)
        else:
            gen_reward_path = generate_initial_reward(gen_path,dataset_name= args.dataset_name)
            gen_reward_model_path = train_reward(gen_reward_path, gen_path)
        
        if args.compute_metrics:
            calculate_metrics_class(gen_path, out_path, iter)
        

        if args.filter:
            gen_path = filter_data(gen_path, gen_reward_model_path, 25000,dataset_name= args.dataset_name)

        if args.fully_synthetic:
            print("Using only self-generated data at each retraining")
            gen_path = gen_path
        else:
            if args.dataset_name == "gaussian8":
                gen_path = mix(args.prop_gen_data, args.original_dataset, gen_path)
            if args.dataset_name == "cifar10":
                gen_path = mix_cifar10(args.prop_gen_data, args.original_dataset, gen_path)
       # break


def get_parser():
    parser = argparse.ArgumentParser(description="Iterative retraining and generation for DDPM model")

    parser.add_argument("--name", type=str, required=True, help="Name of the experiment")
    parser.add_argument("--model_name", type=str, default="ddpm", choices=["ddpm", "edm", "otcfm"],
                        help="Name of the model to use for training")
    parser.add_argument("--dataset_name", type=str, default="cifar10", choices=["gaussian8","cifar10"],
                        help="Name of the model to use for training")
    parser.add_argument("--n_retrain", type=int, default=10, help="Number of retraining iterations")
    parser.add_argument("--num_gen", type=int, default=50000, help="Number of samples to generate")
    parser.add_argument("--num_mimg", type=float, default=0.05, help="Number of million images to train per iteration")

    # parser.add_argument("--network", type=str, required=True, help="Path to the initial network checkpoint")
    parser.add_argument("--network", type=str, default="./models/ddpm/cirfar10/cifar10.pt", help="Path to the initial network checkpoint")
    
    parser.add_argument("--train_dataset", type=str, help="Path to the training dataset directory")
    parser.add_argument("--pregen_dataset", type=str, default="", help="Path to the pregenerated dataset (optional)")
    parser.add_argument("--original_dataset", type=str, default="TODO", help="Path to the original dataset")
    parser.add_argument("--out_path", type=str, default="./output", help="Output directory for saving results")

    parser.add_argument("--prop_gen_data", type=float, default=0.8,
                        help="Proportion of generated data to use in retraining (0 to 1)")
    parser.add_argument("--fully_synthetic", action="store_true",
                        help="Use only synthetic data for retraining")
    parser.add_argument("--compute_metrics", action="store_true",
                        help="Compute metrics after generating samples")
    parser.add_argument("--filter", action="store_true",
                        help="Filter after generating samples")
    parser.add_argument("--malicious", action="store_true",
                        help="Attack or not")
    parser.add_argument("--buget", type=float, default=0.2,
                        help="Attack or not")
    parser.add_argument("--alpha", type=float, default=0.8,
                        help="Attack or not")
    
    parser.add_argument("--random", action="store_true",
                        help="Attack random or not")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training and generation")

    parser.add_argument("--nproc_per_node", type=int, default=1, help="Number of GPUs to use for training")

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    iter_retrain(args)

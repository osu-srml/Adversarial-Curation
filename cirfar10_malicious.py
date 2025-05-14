import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import csv
from functorch import vmap
from cirfar10_reward import load_reward_model, train_reward_model_nosave, reward_loss, reward_loss_h
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.repair import Repair

class DiscreteRepair(Repair):
    def _do(self, problem, X, **kwargs):
        n_var = problem.n_var
        num_nonzero = int(0.2 * n_var)

        for i in range(X.shape[0]):
            indices = np.where(X[i] != 0)[0]

            if len(indices) != num_nonzero:
                # If the non-zero value is not satisfied, re-generate it randomly
                indices = np.random.choice(n_var, size=num_nonzero, replace=False)
                half = len(indices) // 2
                X[i] = 0
                X[i, indices[:half]] = -1
                X[i, indices[half:]] = 1

        return X
    
class DiscreteSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        n_var = problem.n_var
        num_nonzero = int(0.2 * n_var)

        X = np.zeros((n_samples, n_var), dtype=int)

        for i in range(n_samples):
            indices = np.random.choice(n_var, size=num_nonzero, replace=False)
            half = len(indices) // 2
            X[i, indices[:half]] = -1
            X[i, indices[half:]] = 1

        return X


transform = transforms.Normalize(
    mean=(0.4914, 0.4822, 0.4465),  # CIFAR-10 mean
    std=(0.247, 0.243, 0.261)       # CIFAR-10 std
)

class FlippedDataset(Dataset):
    def __init__(self, data, delta):
        self.data = data
        self.delta = delta

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_i, y_i, o_i = self.data[idx]
        return x_i, y_i, o_i + self.delta[idx]

# Dataset
class CIFARRewardDataset(Dataset):
    def __init__(self, data):
        """
        The dataset is used for reward model training and accepts the (x_i, y_i, o_i) data format.。
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_i, y_i, o_i = self.data[idx]
        return (
            x_i.clone().detach().float(),
            y_i.clone().detach().float(),
            torch.tensor(o_i, dtype=torch.float32),
        )

# Load CIFAR-10 data
def load_cifar10_data(root="data", train=True):
    dataset = datasets.CIFAR10(root=root, train=train, transform=transform, download=True)
    data = [(dataset[i][0].numpy(), dataset[j][0].numpy(), 0.5)
            for i, j in zip(range(len(dataset)), range(len(dataset)))]
    return np.array(data, dtype=object)

def load_cifar10_from_fixed_path(data_root):
    """
    Load CIFAR-10 from fixed path

    Parameters
    ----------
    data_root : str
        The root directory of the CIFAR-10 dataset with 'train' and 'test' subdirectories.

    Returns
    -------
    train_loader : DataLoader
    test_loader : DataLoader
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "test")

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader

def flip_labels(data, indices):
    """
    Flips the label of the given index.
    """
    for i in indices:
        x_i, y_i, o_i = data[i]
        data[i] = (x_i, y_i, 1 - o_i)
    return data

def load_data_and_model(data_path, model_path, input_dim):
    reward_data = np.load(data_path, allow_pickle=True)
    r_model = load_reward_model(model_path, input_dim)
    return reward_data, r_model

def train_reward_model_on_flipped_data(dataset, epochs=3, lr=1e-5):
    reward_model = train_reward_model_nosave(dataset)
    return reward_model

def calculate_covariance_gradient(r_model, r_prime_model, data):
    # Stack x_data and y_data from the dataset
    x_data = torch.stack([item[0] for item in data])  # Assume tensors are already on the appropriate device
    y_data = torch.stack([item[1] for item in data])

    # Ensure x_data and y_data are on the same device
    device = x_data.device

    # Generate a random mask on the same device as x_data and y_data
    combined_data = torch.cat((x_data, y_data), dim=0)

    e_r = torch.exp(r_model(combined_data))
    e_r_prime = torch.exp(r_prime_model(combined_data))
    mean_e_r = e_r.mean()
    mean_e_r_prime = e_r_prime.mean()
    cov = ((e_r - mean_e_r) * (e_r_prime - mean_e_r_prime)).mean()

    # Compute gradient with respect to scorer parameters only
    scorer_params = [p for p in r_prime_model.output_layer.parameters() if p.requires_grad]
    grad_cov = torch.autograd.grad(
        outputs=cov,
        inputs=scorer_params,
        create_graph=False,
        allow_unused=False,  # Ensure all scorer parameters are used
    )
    
    grad_cov_flat = torch.cat([g.view(-1) for g in grad_cov])
    return grad_cov_flat
       

def calculate_distance_gradient(r_model, r_prime_model, data):
    """Calculate the gradient of the distance term with respect to model parameters"""
    # Ensure data is loaded as tensors
    # Stack x_data and y_data from the dataset
    x_data = torch.stack([item[0] for item in data])  # Assume tensors are already on the appropriate device
    y_data = torch.stack([item[1] for item in data])

    # Ensure x_data and y_data are on the same device
    device = x_data.device

    # Generate a random mask on the same device as x_data and y_data
    combined_data = torch.cat((x_data, y_data), dim=0)

    reward_r = r_model(combined_data)
    reward_r_prime = r_prime_model(combined_data)

    dist = torch.sum((reward_r - reward_r_prime) ** 2)

    # Compute gradient with respect to scorer parameters only
    scorer_params = [p for p in r_prime_model.output_layer.parameters() if p.requires_grad]
    grad_dist = torch.autograd.grad(
        outputs=dist,
        inputs=scorer_params,
        create_graph=False,
        allow_unused=False,  # Ensure all scorer parameters are used
    )
    grad_dist_flat = torch.cat([g.view(-1) for g in grad_dist])  # Flatten into a single vector
    return grad_dist_flat

def compute_hessian(model, data, loss_fn):
    """Calculate the Hessian matrix of the loss function with respect to the model parameters."""
    # This function should receive the model parameters directly, and the loss function should be correctly wrapped.
    # Collect trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Flatten trainable parameters
    init_params = torch.cat([p.view(-1) for p in trainable_params])

    # Define the wrapped loss function
    def wrapped_loss_fn(model_params):
        idx = 0
        for p in trainable_params:
            numel = p.numel()
            p.data.copy_(model_params[idx:idx + numel].view(p.shape))
            idx += numel

        # Compute the total loss on the dataset
        total_loss = sum(loss_fn(model, x, y, o) for x, y, o in data)
        return total_loss

    # Flatten the initial model parameters
    init_params = torch.cat([p.view(-1) for p in model.parameters() if p.requires_grad])
    # Compute the Hessian using the autograd functionality
    hessian = torch.autograd.functional.hessian(wrapped_loss_fn, init_params)
    return hessian

def second_order_derivative_vectorized(model, data):
    """
    Compute the second-order derivative w.r.t. delta based on the given formula, vectorized.

    Parameters:
        model: nn.Module
            The reward model \( R(\cdot) \).
        data: list of tuples
            Dataset containing pairs \( (x_i, y_i, o_i) \), where \(x_i, y_i \in \mathbb{R}^{D}\).

    Returns:
        grad_grad_delta: torch.Tensor
            Second-order derivative matrix of shape [P, N], where \(P\) is the number of model parameters.
    """
    # Move data to CPU for NumPy compatibility
    x_data = torch.stack([item[0] for item in data]).to(model.device)
    y_data = torch.stack([item[1] for item in data]).to(model.device)

    # Compute R(x) and R(y) for all data points in a single batch
    R_x = model(x_data).squeeze()  # Shape: [N]
    R_y = model(y_data).squeeze()  # Shape: [N]

    output_layer_params = [p for p in model.output_layer.parameters() if p.requires_grad]

    # Compute the total loss as a vector for all samples
    loss_per_sample = R_x - R_y  # Shape: [N]
    # Compute gradients for each sample separately
    per_sample_grads = []
    for i in range(len(loss_per_sample)):
        grad = torch.autograd.grad(
            outputs=loss_per_sample[i],  # Scalar loss for one sample
            inputs=output_layer_params,
            retain_graph=True,  # Keep graph for further gradients
            create_graph=False,  # Do not create higher-order gradients
            allow_unused=False,
        )
        # Flatten and concatenate gradients for this sample
        grad_flat = torch.cat([g.view(-1) for g in grad], dim=0)
        per_sample_grads.append(grad_flat)  # Append flattened gradients

    # Stack gradients into a single tensor
    per_sample_grads_matrix = torch.stack(per_sample_grads, dim=1)  # Shape: [P, N]

    return per_sample_grads_matrix

def compute_single_sample_grad(model, x, y):
    model.eval()
    x = x.unsqueeze(0)  # Add batch dimension
    y = y.unsqueeze(0)  # Add batch dimension

    R_x = model(x).squeeze()
    R_y = model(y).squeeze()
    loss = R_x - R_y  # Scalar loss
    output_layer_params = [p for p in model.output_layer.parameters() if p.requires_grad]
    grad = torch.autograd.grad(
        outputs=loss,
        inputs=output_layer_params,
        retain_graph=True,
        create_graph=False,
        allow_unused=False,
    )
    model.train()
    return torch.cat([g.view(-1) for g in grad], dim=0)

def second_order_derivative_vmap(model, data):
    x_data = torch.stack([item[0] for item in data]).to(model.device)
    y_data = torch.stack([item[1] for item in data]).to(model.device)

    # Vectorize the gradient computation
    grads_matrix = vmap(compute_single_sample_grad, in_dims=(None, 0, 0))(model, x_data, y_data)
    return grads_matrix.T  # Transpose to [P, N]

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

def save_data_pairs(output_path, original_data, device):
    """
    Save data pairs (class1, class2, o_i) using VGG11 model for classification.

    Parameters:
        output_path: str
            Path to save the data pairs.
        original_data: list
            List of tuples (x_i, y_i, o_i).
        device: torch.device
            Device for computation (CPU or GPU).
    """
    batch_size =512
    # Load VGG11 model
    vgg11_model = load_vgg11_model().to(device)
    vgg11_model.eval()

    parent_path = os.path.dirname(output_path)
    delta_path = os.path.join(parent_path, "delta")
    os.makedirs(delta_path, exist_ok=True)

    # Prepare output arrays for class predictions
    class1_labels = []
    class2_labels = []

    # Process data in batches
    for i in range(0, len(original_data), batch_size):
        batch_data = original_data[i:i + batch_size]
        x_batch = torch.stack([torch.tensor(x_i, dtype=torch.float32) for x_i, _, _ in batch_data]).to(device)
        y_batch = torch.stack([torch.tensor(y_i, dtype=torch.float32) for _, y_i, _ in batch_data]).to(device)

        with torch.no_grad():
            class1_predictions = vgg11_model(x_batch)
            class2_predictions = vgg11_model(y_batch)

            class1_labels.extend(torch.argmax(class1_predictions, dim=1).cpu().numpy())
            class2_labels.extend(torch.argmax(class2_predictions, dim=1).cpu().numpy())

    # Save data pairs to file
    output_file = os.path.join(delta_path, "data_pairs.csv")
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(["Index", "Class1", "Class2", "o_i"])

        for i, (_, _, o_i) in enumerate(original_data):
            writer.writerow([i, class1_labels[i], class2_labels[i], o_i])

    print(f"Data pairs saved to: {output_file}")

def save_delta_values(output_path, delta, epoch):
    """
    Save delta values for each epoch.

    Parameters:
        output_path: str
            Original output path used to derive the parent path.
        delta: torch.Tensor
            Tensor containing delta values.
        epoch: int
            Current epoch number.
    """
    # Create the parent path + delta directory
    parent_path = os.path.dirname(output_path)
    delta_path = os.path.join(parent_path, "delta")
    os.makedirs(delta_path, exist_ok=True)

    # Generate file name dynamically based on the epoch
    output_file = os.path.join(delta_path, f"delta_values_epoch_{epoch}.csv")

    # Save delta values to file
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(["Index", "Delta"])

        for i, delta_value in enumerate(delta):
            writer.writerow([i, delta_value.item()])

    print(f"Delta values for epoch {epoch} saved to: {output_file}")


def attack_white(data_path, model_path, gen_path, B, alpha1, alpha2, epochs=5, lr=1e-4):
    print("Attacking")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu = torch.device("cpu")

    # Load dataset into memory on GPU initially
    original_data = np.load(data_path, allow_pickle=True)

    processed_data = []
    for x_i, y_i, o_i in original_data:
        x_transformed = transform(torch.tensor(x_i, dtype=torch.float32))
        y_transformed = transform(torch.tensor(y_i, dtype=torch.float32))
        processed_data.append((x_transformed, y_transformed, o_i))

    dataset_length = len(original_data)
    budget = int(B * dataset_length)

    # Prepare the dataset in memory on GPU
    data = [(x.to(device), y.to(device), torch.tensor(o, dtype=torch.float32, device=device))
            for x, y, o in processed_data]

    input_dim = 3 * 32 * 32  # CIFAR-10 input dimension
    r_model = load_reward_model(model_path).to(device)

    # Initialize delta based on o_i values
    delta = torch.tensor([1 if o == 0 else -1 for _, _, o in processed_data], dtype=torch.float32, device=device, requires_grad=True)
    #delta_accumulated = torch.zeros(len(original_data), device=device)
    delta_accumulated = delta.clone().detach()

    # Process in batches
    batch_size = 16384
    total_batches = (len(data) // batch_size) * epochs
    with tqdm(total=total_batches, desc="Training Progress") as pbar:
        for step in range(epochs):
            # print(f"Epoch {step + 1}")

            # Generate flipped_data and train r_model_prime
            flipped_data = [(x_i, y_i, o_i + delta[i]) for i, (x_i, y_i, o_i) in enumerate(data)]
            r_model_prime = train_reward_model_nosave(flipped_data).to(device)

            # Free memory used by flipped_data
            del flipped_data
            torch.cuda.empty_cache()

            # Move data to CPU to free GPU memory before batch processing
            data = [(x.cpu(), y.cpu(), o.cpu()) for x, y, o in data]
            delta = delta.cpu()
            delta_accumulated = delta_accumulated.cpu()
            
            for batch_start in range(0, len(data), batch_size):

                # Load the batch back to GPU
                batch_end = batch_start + batch_size
                batch_data = data[batch_start:batch_end]
                batch_data_gpu = [(x.to(device), y.to(device), o.to(device)) for x, y, o in batch_data]
                batch_delta = delta[batch_start:batch_end].to(device)

                # Compute gradients
                grad_cov = calculate_covariance_gradient(r_model, r_model_prime, batch_data_gpu)
                grad_dist = calculate_distance_gradient(r_model, r_model_prime, batch_data_gpu)
                grad_total = alpha1 * grad_cov + alpha2 * grad_dist

                # Compute Hessian
                hessian = compute_hessian(r_model_prime, batch_data_gpu, reward_loss_h)
                regularization = 1e-5 * torch.eye(hessian.size(0), device=hessian.device)
                hessian_inv = torch.linalg.pinv(hessian + regularization)

                # Compute second-order derivatives
                grad_grad_delta = second_order_derivative_vectorized(r_model_prime, batch_data_gpu)
              
                # Calculate result for the batch
                result = grad_total @ (-hessian_inv @ grad_grad_delta)

                # Normalize and update delta for the batch
                epsilon = 1e-8
                norm_result = result.norm() + epsilon
                grad_normalized = result / norm_result
                step_size = 1
                batch_delta = grad_normalized * step_size

                # Update delta with the new values for the batch
                delta[batch_start:batch_end] = batch_delta.cpu().detach()
                # Move accumulated delta back to CPU
                delta_accumulated[batch_start:batch_end] += batch_delta.cpu().detach()


                # Free GPU memory for this batch
                del grad_cov, grad_dist, grad_total, hessian, hessian_inv, grad_grad_delta, result,  grad_normalized, batch_data_gpu, batch_delta
                torch.cuda.empty_cache()

                pbar.update(1)

            # Move delta back to GPU for the next epoch
            delta = delta.to(device)
            delta_accumulated = delta_accumulated.to(device)

            # Clear variables for the current epoch
            del r_model_prime
            torch.cuda.empty_cache()

            #save_delta_values(gen_path, delta, step)

    delta_mean = delta_accumulated / epochs
    _, top_indices = torch.topk(delta_mean.abs(), budget)
    final_flipped_data = flip_labels(original_data, top_indices.cpu().numpy())

    reward_dir = os.path.join(os.path.dirname(gen_path), "malicious_reward")
    os.makedirs(reward_dir, exist_ok=True)
    reward_path = os.path.join(reward_dir, "reward_dataset.npy")
    np.save(reward_path, final_flipped_data)
    return reward_path

def calculate_global_distance(r_model, r_model_prime, flipped_data, batch_size=5120):
    device = next(r_model.parameters()).device
    total_dist = 0.0
    num_batches = 0

    for i in range(0, len(flipped_data), batch_size):
        batch = flipped_data[i:i+batch_size]
        x_batch = torch.stack([x for x, _, _ in batch]).to(device)
        y_batch = torch.stack([y for _, y, _ in batch]).to(device)

        output_r = r_model(x_batch)
        output_r_prime = r_model_prime(x_batch)
        dist = torch.norm(output_r - output_r_prime, p=2).mean().item()

        total_dist += dist
        num_batches += 1
    return total_dist / num_batches


def calculate_global_covariance(r_model, r_model_prime, flipped_data, batch_size=5120):
    device = next(r_model.parameters()).device
    total_cov = 0.0
    num_batches = 0

    for i in range(0, len(flipped_data), batch_size):
        batch = flipped_data[i:i+batch_size]
        x_batch = torch.stack([x for x, _, _ in batch]).to(device)
        y_batch = torch.stack([y for _, y, _ in batch]).to(device)

 
        output_r = r_model(x_batch)
        output_r_prime = r_model_prime(x_batch)

 
        mean_r = output_r.mean(dim=0)
        mean_r_prime = output_r_prime.mean(dim=0)
        cov = ((output_r - mean_r) * (output_r_prime - mean_r_prime)).mean().item()

        total_cov += cov
        num_batches += 1

    return total_cov / num_batches


def attack_random(data_path, gen_path, B):
    print("Random attacking")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset into memory
    original_data = np.load(data_path, allow_pickle=True)

    dataset_length = len(original_data)
    budget = int(B * dataset_length)

    top_indices = torch.randperm(dataset_length)[:budget]

    final_flipped_data = flip_labels(original_data, top_indices.cpu().numpy())

    reward_dir = os.path.join(os.path.dirname(gen_path), "malicious_reward")
    os.makedirs(reward_dir, exist_ok=True)
    reward_path = os.path.join(reward_dir, "reward_dataset.npy")
    np.save(reward_path, final_flipped_data)
    
    return reward_path

def attack_pareto(data_path, model_path, gen_path, B, alpha1, alpha2, pop_size=50, n_gen=2):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_data = np.load(data_path, allow_pickle=True)
    processed_data = []
    for x_i, y_i, o_i in original_data:
        x_transformed = transform(torch.tensor(x_i, dtype=torch.float32))
        y_transformed = transform(torch.tensor(y_i, dtype=torch.float32))
        processed_data.append((x_transformed.to(device), y_transformed.to(device), o_i))
    
    dataset_length = len(original_data)
    budget = int(B * dataset_length)
    data = [(x.to(device), y.to(device), torch.tensor(o, dtype=torch.float32, device=device))
            for x, y, o in processed_data]

    r_model = load_reward_model(model_path).to(device)
    
    class ParetoProblem(Problem):
        def __init__(self, processed_data, r_model, device):
            super().__init__(n_var=len(processed_data), n_obj=2, xl=-1.0, xu=1.0)
            self.processed_data = processed_data
            self.r_model = r_model
            self.device = device
            self.current_gen = 0 

            self.cov_history = []
            self.dist_history = []

        def _evaluate(self, X, out, *args, **kwargs):

            self.current_gen += 1
            print(f"=== Start of Generation {self.current_gen} ===")

            cov_losses = []
            dist_losses = []

            for i in range(X.shape[0]):
                delta = X[i]
                print(f"  Processing sample {i + 1} in Generation {self.current_gen}")

                flipped_data = [(x, y, torch.clamp(o + delta[j], min=-1, max=1))
                for j, (x, y, o) in enumerate(self.processed_data)]

                r_model_prime = train_reward_model_nosave(flipped_data).to(device)

                cov_loss = calculate_global_covariance(r_model, r_model_prime, flipped_data, batch_size=5120)
                dist_loss = calculate_global_distance(r_model, r_model_prime, flipped_data, batch_size=5120)

                cov_losses.append(cov_loss)
                dist_losses.append(dist_loss)

            self.cov_history.append(cov_losses)
            self.dist_history.append(dist_losses)

            out["F"] = np.column_stack([cov_losses, dist_losses])
    
    problem = ParetoProblem(data, r_model, device)
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=DiscreteSampling(),
        repair=DiscreteRepair()
    )

    res = minimize(problem, algorithm, termination=('n_gen', n_gen), seed=1, verbose=True)

    pareto_inputs_fixed = np.clip(np.round(res.X), -1, 1)
    
    pareto_solutions = res.F
    pareto_inputs = res.X
    
    best_delta = pareto_inputs_fixed[np.argmin(res.F[:, 0] + res.F[:, 1])]

    updated_delta = [
        torch.clamp( best_delta[j] + o_i, min=-1, max=1) - o_i
        for j, (_, _, o_i) in enumerate(data)
    ]

    top_indices = [j for j, d in enumerate(updated_delta) if d != 0]
    top_indices_np = np.array(top_indices)
    final_flipped_data = flip_labels(original_data, top_indices_np)

    reward_dir = os.path.join(os.path.dirname(gen_path), "malicious_reward")
    os.makedirs(reward_dir, exist_ok=True)
    reward_path = os.path.join(reward_dir, "reward_dataset.npy")
    np.save(reward_path, final_flipped_data)
    return reward_path


def attack_dist(data_path, model_path, gen_path, B):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu = torch.device("cpu")

    # Load dataset into memory on GPU initially
    original_data = np.load(data_path, allow_pickle=True)

    processed_data = []
    for x_i, y_i, o_i in original_data:
        x_transformed = transform(torch.tensor(x_i, dtype=torch.float32))
        y_transformed = transform(torch.tensor(y_i, dtype=torch.float32))
        processed_data.append((x_transformed, y_transformed, o_i))

    dataset_length = len(original_data)
    budget = int(B * dataset_length)

    # Prepare the dataset in memory on GPU
    data = [(x.to(device), y.to(device), torch.tensor(o, dtype=torch.float32, device=device))
            for x, y, o in processed_data]

    input_dim = 3 * 32 * 32  # CIFAR-10 input dimension
    r_model = load_reward_model(model_path).to(device)
    
    batch_size=512
    dist = torch.empty(len(data))
    for batch_start in range(0, len(data), batch_size):
        # Load the batch back to GPU
        batch_end = batch_start + batch_size
        batch_data = data[batch_start:batch_end]
        #        batch_data_gpu = [(x.to(device), y.to(device), o.to(device)) for x, y, o in batch_data]
    
    
        x_data = torch.stack([item[0] for item in batch_data])  # Assume tensors are already on the appropriate device
        y_data = torch.stack([item[1] for item in batch_data])

        reward_x = r_model(x_data)
        reward_y = r_model(y_data)

        dist_batch = torch.abs(reward_x - reward_y)
        dist[batch_start:batch_end] = dist_batch.cpu().detach()

    _, top_indices = torch.topk(dist, budget)
    final_flipped_data = flip_labels(original_data, top_indices.cpu().numpy())

    reward_dir = os.path.join(os.path.dirname(gen_path), "malicious_reward")
    os.makedirs(reward_dir, exist_ok=True)
    reward_path = os.path.join(reward_dir, "reward_dataset.npy")
    np.save(reward_path, final_flipped_data)
    return reward_path

def attack_top(data_path, model_path, gen_path, B):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    original_data = np.load(data_path, allow_pickle=True)
    dataset_length = len(original_data)
    budget = int(B * dataset_length)  # Number of pairs to flip
    batch_size=512

    # Prepare data for batch processing
    processed_data = []
    x_list, y_list = [], []
    for x_i, y_i, o_i in original_data:
        x_tensor = torch.tensor(x_i, dtype=torch.float32)
        y_tensor = torch.tensor(y_i, dtype=torch.float32)
        processed_data.append((x_tensor, y_tensor, o_i))
        x_list.append(x_tensor)
        y_list.append(y_tensor)

    # Stack tensors for batch processing
    x_tensor_batch = torch.stack(x_list).to(device)
    y_tensor_batch = torch.stack(y_list).to(device)

    # Load reward model
    r_model = load_reward_model(model_path).to(device)

    # Compute rewards in batches
    rewards_x, rewards_y = [], []
    for batch_start in range(0, dataset_length, batch_size):
        batch_data = processed_data[batch_start:batch_start + batch_size]
        x_batch = torch.stack([item[0] for item in batch_data]).to(device)
        y_batch = torch.stack([item[1] for item in batch_data]).to(device)

        with torch.no_grad():
            rewards_x_batch = r_model(x_batch).squeeze().cpu()
            rewards_y_batch = r_model(y_batch).squeeze().cpu()

        rewards_x.append(rewards_x_batch)
        rewards_y.append(rewards_y_batch)

    # Concatenate rewards
    rewards_x = torch.cat(rewards_x)  # Rewards for x
    rewards_y = torch.cat(rewards_y)  # Rewards for y

    # Combine all rewards into a single list with indices
    rewards_with_indices = []
    for i, (r_x, r_y) in enumerate(zip(rewards_x, rewards_y)):
        rewards_with_indices.append((r_x.item(), i, "x"))  # Reward from x
        rewards_with_indices.append((r_y.item(), i, "y"))  # Reward from y

    # Sort rewards in descending order
    rewards_with_indices.sort(key=lambda item: -item[0])  # Sort by reward value (descending)

    # Collect indices of flipped pairs
    flipped_indices = set()  # To ensure no duplicate flips
    top_indices = []
    flips = 0

    for reward, pair_idx, z_type in rewards_with_indices:
        if flips >= budget:
            break
        if pair_idx not in flipped_indices:
            top_indices.append(pair_idx)
            flipped_indices.add(pair_idx)
            flips += 1

    # Use the flip_labels function to apply flips
    final_flipped_data = flip_labels(original_data, top_indices)

    # Save the modified dataset
    reward_dir = os.path.join(os.path.dirname(gen_path), "malicious_reward")
    os.makedirs(reward_dir, exist_ok=True)
    reward_path = os.path.join(reward_dir, "reward_dataset.npy")
    np.save(reward_path, final_flipped_data)

    print(f"Attack completed. Flipped {flips} data pairs. Modified dataset saved to {reward_path}")
    return reward_path
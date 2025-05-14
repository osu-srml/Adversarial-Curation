import torch
import numpy as np
from reward import load_reward_model, train_reward_model_nosave, reward_loss
import os
from functorch import vmap, jacrev

def load_data_and_model(data_path, model_path, input_dim):
    reward_data = np.load(data_path, allow_pickle=True)
    r_model = load_reward_model(model_path, input_dim)
    return reward_data, r_model

def train_reward_model_on_flipped_data(dataset, input_dim, epochs=5, lr=1e-4):
    reward_model = train_reward_model_nosave(dataset)
    return reward_model

def calculate_covariance_gradient(r_model, r_prime_model, data):
    # Ensure data is loaded as tensors
    x_data = torch.stack([item[0] for item in data])  # Use torch.stack to handle tensor dimensions correctly
    y_data = torch.stack([item[1] for item in data])

    combined_data = torch.cat([x_data, y_data], dim=0)  # Concatenate along a suitable dimension

    # Forward pass through models
    e_r = torch.exp(r_model(combined_data))
    e_r_prime = torch.exp(r_prime_model(combined_data))

    # Compute means
    mean_e_r = e_r.mean()
    mean_e_r_prime = e_r_prime.mean()

    # Compute covariance
    cov = ((e_r - mean_e_r) * (e_r_prime - mean_e_r_prime)).mean()

    # Compute gradient
    grad_cov = torch.autograd.grad(outputs=cov, inputs=list(r_model.parameters()), create_graph=True)
 
    grad_cov_flat = torch.cat([g.view(-1) for g in grad_cov])  # Flatten gradients into a vector
    # Print the shape of grad_cov_flat
    # print("Shape of grad_cov_flat:", grad_cov_flat.shape)
    return grad_cov_flat
       

def calculate_distance_gradient(r_model, r_prime_model, data):
    """Calculate the gradient of the distance term with respect to model parameters"""
    # Ensure data is loaded as tensors
    x_data = torch.stack([item[0] for item in data])  # Use torch.stack to handle tensor dimensions correctly
    y_data = torch.stack([item[1] for item in data])

    combined_data = torch.cat([x_data, y_data], dim=0)  # Concatenate along a suitable dimension

    reward_r = r_model(combined_data)
    reward_r_prime = r_prime_model(combined_data)

    dist = torch.sum((reward_r - reward_r_prime) ** 2)

    # Compute the gradient as a vector
    grad_dist = torch.autograd.grad(
        outputs=dist,
        inputs=list(r_model.parameters()),  # Gradient w.r.t. model parameters
        create_graph=True
    )
    grad_dist_flat = torch.cat([g.view(-1) for g in grad_dist])  # Flatten into a single vector
    return grad_dist_flat

def flip_labels(data, indices):
    if isinstance(data, np.ndarray) and data.dtype == np.object_:
        data = data.tolist()

    for i in indices:
        x_i, y_i, o_i = data[i]
        data[i] = (x_i, y_i, 1 - o_i)  # Flip the label o_i
    
    return np.array(data, dtype=object)

def compute_hessian(model, data, loss_fn):
    """Calculate the Hessian matrix of the loss function with respect to the model parameters."""
    # This function should receive the model parameters directly, and the loss function should be correctly wrapped.
    def wrapped_loss_fn(model_params):
        # Load the current parameters into the model
        idx = 0
        for p in model.parameters():
            numel = p.numel()
            p.data.copy_(model_params[idx:idx + numel].reshape(p.shape))
            idx += numel
        
        # Calculate the total loss on the dataset
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
    # Extract and stack x_data and y_data from data
    x_data = torch.tensor(np.stack([item[0] for item in data]), dtype=torch.float32, requires_grad=True)  # Shape: [N, D]
    y_data = torch.tensor(np.stack([item[1] for item in data]), dtype=torch.float32, requires_grad=True)  # Shape: [N, D]

    # Compute R(x) and R(y) for all data points in a single batch
    R_x = model(x_data).squeeze()  # Shape: [N]
    R_y = model(y_data).squeeze()  # Shape: [N]

    # Compute gradients w.r.t. model parameters for R(x) and R(y)
    grad_R_x = torch.autograd.grad(
        outputs=R_x.sum(), inputs=list(model.parameters()), create_graph=True, retain_graph=True
    )
    grad_R_y = torch.autograd.grad(
        outputs=R_y.sum(), inputs=list(model.parameters()), create_graph=True, retain_graph=True
    )

    # Flatten gradients into a single vector for each model parameter
    grad_R_x_flat = torch.cat([g.view(-1) for g in grad_R_x], dim=0)  # Shape: [P]
    grad_R_y_flat = torch.cat([g.view(-1) for g in grad_R_y], dim=0)  # Shape: [P]

    # Compute exp(R(x)) and exp(R(y))
    exp_R_x = torch.exp(R_x)  # Shape: [N]
    exp_R_y = torch.exp(R_y)  # Shape: [N]

    # Compute denominator: e^(R(x)) + e^(R(y)) (Shape: [N])
    denom = exp_R_x + exp_R_y

    # Compute weighted gradients
    weighted_grad_x = (exp_R_x / denom).unsqueeze(0) * grad_R_x_flat.unsqueeze(1)  # Shape: [P, N]
    weighted_grad_y = (exp_R_y / denom).unsqueeze(0) * grad_R_y_flat.unsqueeze(1)  # Shape: [P, N]

    # Compute the second-order derivative matrix
    second_derivative = (
        -grad_R_x_flat.unsqueeze(1)
        - grad_R_x_flat.unsqueeze(1)
        + 2 * (weighted_grad_x + weighted_grad_y)
    )  # Shape: [P, N]

    return second_derivative

def attack_white(data_path, model_path, gen_path, B, alpha1, alpha2, epochs=5, lr=1e-4):
    original_data = np.load(data_path, allow_pickle=True)
    x_i_sample = original_data[0][0]
    input_dim = x_i_sample.shape[0]
    r_model = load_reward_model(model_path, input_dim)
    
    delta = torch.zeros(len(original_data), requires_grad=True)

    # Convert data to tensors
    data = [(torch.tensor(x_i, dtype=torch.float32), torch.tensor(y_i, dtype=torch.float32), torch.tensor(o_i, dtype=torch.float32)) for x_i, y_i, o_i in original_data]

    delta = torch.zeros(len(data), requires_grad=True)
    delta_accumulated = torch.zeros(len(data))

    for step in range(epochs):
        print(f"Epoch {step + 1}")
        # Generate flipped dataset based on current delta
        flipped_data = [(x_i, y_i, o_i + d) for (x_i, y_i, o_i), d in zip(data, delta)]

        # Retrain the r_model_prime on the flipped dataset
        r_model_prime = train_reward_model_on_flipped_data(flipped_data, input_dim, epochs, lr=1e-5)
        # Calculate gradients of covariance and distance
        grad_cov = calculate_covariance_gradient(r_model, r_model_prime, data)
        grad_dist = calculate_distance_gradient(r_model, r_model_prime, data)

        # Combine gradients
        grad_total = alpha1 * grad_cov + alpha2 * grad_dist
        print("Shape of grad_total:", grad_total.shape)  # Expected to be (P,)

        # Compute Hessian of the loss with respect to theta and invert
        hessian = compute_hessian(r_model, data, reward_loss)
        hessian_inv = torch.linalg.pinv(hessian)  # Pseudo-inverse for numerical stability
        print("Shape of hessian_inv:", hessian_inv.shape)  # Expected to be (P, P)

        # Compute the gradient of the gradient with respect to delta
        grad_grad_delta = second_order_derivative_vectorized(r_model, data)
        print("Shape of grad_grad_delta:", grad_grad_delta.shape)  # Expected to be (P, N)

        # Calculate result as grad_total * -hessian_inv * grad_grad_delta
        result = grad_total @ (-hessian_inv @ grad_grad_delta)
        print("Shape of result:", result.shape)  # Expected to be (N,)

        # Normalize the gradient (for the update rule in the image)
        grad_normalized = result / result.norm()

        # Update delta based on the normalized gradient and step size
        step_size = 1
        delta = grad_normalized * step_size
        for i, (_, _, o_i) in enumerate(data):
            max_delta = 1 - o_i  # Maximum delta so that o_i + delta_i <= 1
            min_delta = -o_i     # Minimum delta so that o_i + delta_i >= 0
            delta[i] = torch.clamp(delta[i], min=min_delta, max=max_delta)


        # Accumulate delta updates
        delta_accumulated += delta.detach()

    # Compute the average delta over all epochs
    delta_mean = delta_accumulated / epochs    # Determine indices to flip based on final delta values
    _, top_indices = torch.topk(delta_mean.abs(), B)
    final_flipped_data = flip_labels(original_data, top_indices.numpy())

    reward_dir = os.path.join(os.path.dirname(gen_path), "malicious_reward")
    os.makedirs(reward_dir, exist_ok=True)

    reward_path = os.path.join(reward_dir, "reward_dataset.npy")
    np.save(reward_path, final_flipped_data)
    return reward_path
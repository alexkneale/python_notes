# ============================================================================
# PYTORCH DEEP DIVE: CUSTOM MODULES, INITIALIZATION, AND ADVANCED TRAINING
# ============================================================================
# Target Audience: Senior ML Engineers / Researchers (MSc/PhD level).
# Topic: Custom Layers, Buffers vs Parameters, Weight Init Theory, Custom Focal Loss,
#        Gradient Clipping, Mixed Precision (AMP), L2 Regularization vs AdamW.
# Compatibility: Python 3.10+, PyTorch 2.0+
# ============================================================================

# ============================================================================
# BEGINNER PRIMER: BASIC CONCEPTS
# ============================================================================
"""
1. What is a Neural Network Layer?
   - A layer is a functional block that transforms input data. It contains parameters 
     called weights (multipliers) and biases (offsets). For input X, a linear layer computes 
     Y = X * W + b.
2. What is a Loss Function?
   - A loss function (or cost function) measures how poorly our model is performing. 
     It compares predicted outputs with the ground truth and returns a single penalty score. 
     Lower loss values correspond to more accurate predictions.
3. What is an Optimizer?
   - An optimizer is the algorithm (like Stochastic Gradient Descent - SGD, or Adam) 
     responsible for updating the model's weights using calculated gradients. It determines 
     how large of a step to take (learning rate) to navigate down the loss curve.
4. What is Mixed Precision (AMP)?
   - Normally, numbers are stored in 32-bit floats. Mixed precision uses lightweight 
     16-bit floats for heavy matrix math (speeding up training and saving GPU memory) 
     while retaining 32-bit floats for critical parts to maintain numerical stability.
"""



import torch
import torch.nn as nn
import torch.optim as optim
import math
import time

# ============================================================================
# 1. CUSTOM nn.Module: PARAMETERS, BUFFERS, AND CUSTOM LAYERS
# ============================================================================
"""
When subclassing nn.Module, understanding how PyTorch tracks state is crucial:
1. nn.Parameter: Tensors that are automatically registered as model parameters 
   and updated by optimizers (requires_grad=True by default).
2. Buffers: Model state that must be saved/loaded with state_dict but is NOT 
   a parameter updated by backpropagation (e.g., running mean and variance 
   in BatchNorm). Registered via `register_buffer(name, tensor)`.
3. Standard instance attributes: Regular Tensors or Python values assigned to 
   `self` will NOT be registered, causing state to be lost during serialization.
"""

class CustomLinear(nn.Module):
    """
    Custom fully-connected layer mimicking nn.Linear, but showing raw 
    weight/bias management, registration, and math under the hood.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Raw allocation of weights and bias as nn.Parameter
        # We initialize them as empty/zero; initialization logic belongs elsewhere!
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        
        # Demonstrating a model Buffer - tracking the count of forward passes
        self.register_buffer("forward_passes", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Increment our buffer. Note: we increment in-place.
        # Since forward_passes is a buffer, it will be saved in state_dict.
        self.forward_passes += 1
        
        # Linear layer forward equation: y = xW^T + b
        # We use F.linear or torch.addmm for high-performance matrix multiplication.
        return torch.addmm(self.bias, x, self.weight.t())

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


# ============================================================================
# 2. WEIGHT INITIALIZATION THEORY (GLOROT VS KAIMING)
# ============================================================================
"""
Why is weight initialization critical?
During forward propagation, activations can explode or collapse to zero.
During backward propagation, gradients can explode or vanish.

A. GLOROT / XAVIER INITIALIZATION (Designed for Symmetric Activations: Tanh, Sigmoid)
   Assumptions: Activations are linear around zero.
   To maintain constant variance of activations across layers, we set:
     Var(W) = 2 / (fan_in + fan_out)
   - Xavier Normal: W ~ N(0, std^2) where std = sqrt(2 / (fan_in + fan_out))
   - Xavier Uniform: W ~ U(-a, a) where a = sqrt(6 / (fan_in + fan_out))

B. HE / KAIMING INITIALIZATION (Designed for Non-Symmetric Activations: ReLU, LeakyReLU)
   ReLU cuts off negative values, halving activation variance. To compensate:
     Var(W) = 2 / fan_in (for forward pass preservation)
   - Kaiming Normal: W ~ N(0, std^2) where std = sqrt(2 / fan_in)
"""

def demonstrate_initialization_importance():
    print("\n" + "="*80)
    print("2. WEIGHT INITIALIZATION THEORY & EMPIRICAL PROOF")
    print("="*80)

    fan_in = 512
    depth = 50
    x_input = torch.randn(128, fan_in) # Batch size = 128

    print("Simulating a deep neural network forward pass (50 layers)...")

    # 1. Bad Initialization: Random standard normal weights scaled by 1.0 (Too large!)
    x = x_input.clone()
    for _ in range(depth):
        w = torch.randn(fan_in, fan_in) * 1.0
        x = torch.relu(torch.matmul(x, w))
    print(f"  [Bad Init (std=1.0)] Activation mean: {x.mean().item():.4f}, std: {x.std().item():.4f} (Exploded!)")

    # 2. Bad Initialization: Weights initialized too small (std=0.01)
    x = x_input.clone()
    for _ in range(depth):
        w = torch.randn(fan_in, fan_in) * 0.01
        x = torch.relu(torch.matmul(x, w))
    print(f"  [Bad Init (std=0.01)] Activation mean: {x.mean().item():.4f}, std: {x.std().item():.4f} (Vanished!)")

    # 3. Proper Kaiming (He) Initialization for ReLU
    x = x_input.clone()
    for i in range(depth):
        # Var(W) = 2 / fan_in => std = sqrt(2/fan_in)
        std = math.sqrt(2.0 / fan_in)
        w = torch.randn(fan_in, fan_in) * std
        x = torch.relu(torch.matmul(x, w))
    print(f"  [Kaiming Init]        Activation mean: {x.mean().item():.4f}, std: {x.std().item():.4f} (Perfect stability!)")


# ============================================================================
# 3. CUSTOM LOSS FUNCTION: THE MATHEMATICALLY STABLE FOCAL LOSS
# ============================================================================
"""
Focal Loss is used to solve extreme class imbalance (e.g. 1 foreground class vs 1000 background classes).
It downweights easy-to-classify examples, focusing the model on hard negative cases.

Binary Focal Loss Equation:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
where:
    p_t = p if y = 1 else (1 - p)
    alpha_t = alpha if y = 1 else (1 - alpha)

To ensure numerical stability in PyTorch, we should formulate this in terms of 
logits (logits x, rather than sigmoid outputs p) to prevent taking log(0) or 
suffering precision truncation underflow.

Log-logits formulation:
    p = sigmoid(x) = 1 / (1 + e^-x)
    log(p) = -softplus(-x)
    log(1 - p) = -softplus(x)
"""

class StableBinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: Unnormalized logits from the network (shape matching targets)
        targets: Binary targets (0 or 1)
        """
        # Element-wise binary cross entropy terms computed stably using softplus
        # log(p) = -softplus(-logits)
        log_p_1 = -nn.functional.softplus(-logits)
        # log(1-p) = -softplus(logits)
        log_p_0 = -nn.functional.softplus(logits)

        # Get the positive exponent terms: (1 - p)^gamma = exp(gamma * log(1-p))
        p_t_term = torch.exp(self.gamma * (log_p_0 if targets == 1 else log_p_1))

        # Weighting factor alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Naive cross-entropy component
        bce = targets * log_p_1 + (1 - targets) * log_p_0

        # Loss formula: -alpha_t * (1 - p_t)^gamma * log(p_t)
        loss = -alpha_t * p_t_term * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ============================================================================
# 4. DECOUPLED WEIGHT DECAY: L2 REGULARIZATION VS ADAMW
# ============================================================================
"""
An important conceptual question in ML interviews is: 
"What is the difference between adding L2 regularization to the loss vs using AdamW?"

1. Naive L2 regularization adds a penalty directly to the objective function:
     Loss_reg(theta) = Loss(theta) + (lambda / 2) * ||theta||^2
   The gradient becomes:
     g_t = grad_theta(Loss) + lambda * theta

2. For Standard SGD, this is mathematically identical to updating:
     theta_{t+1} = theta_t - lr * (g_t_loss + lambda * theta_t)
                 = (1 - lr * lambda) * theta_t - lr * g_t_loss  (Weight Decay!)

3. However, for adaptive optimizers like Adam, the gradient is scaled by its historical 
   moving average (s_t):
     theta_{t+1} = theta_t - (lr / (sqrt(s_t) + eps)) * (g_t_loss + lambda * theta_t)
   This means parameters with large gradients (hence large s_t) get LESS L2 regularization! 
   This violates the objective of penalizing weights uniformly.

4. AdamW (Decoupled Weight Decay) decouples the regularizer from the gradient updates:
     theta_{t+1} = theta_t - lr * lambda * theta_t - (lr / (sqrt(s_t) + eps)) * g_t_loss
   The decay is applied directly to the parameters, restoring the proper decay.
"""


# ============================================================================
# 5. HIGH-PERFORMANCE TRAINING LOOP WITH MIXED PRECISION (AMP)
# ============================================================================
"""
To train modern deep learning models efficiently, we leverage:
1. Automatic Mixed Precision (AMP): Performs matrix multiplications in float16 (or bfloat16) 
   while keeping critical steps (like loss scaling and parameter accumulation) in float32.
   Uses a GradScaler to dynamically adjust scaling of the loss, preventing underflow 
   of small gradients.
2. Gradient Clipping: Prevents exploding gradients by scaling the norm of gradients 
   across all parameters combined.
"""

def train_focal_loss_model_with_amp():
    print("\n" + "="*80)
    print("5. HIGH-PERFORMANCE TRAINING LOOP WITH AUTOMATIC MIXED PRECISION (AMP)")
    print("="*80)

    # 1. Synthetic data setup for binary classification with severe class imbalance (1% positives)
    N_samples = 5000
    features = 64
    X = torch.randn(N_samples, features)
    # Target is generated such that only a small portion is class 1
    y = (X[:, 0] + X[:, 1] * 2.0 > 4.5).float().unsqueeze(1)
    print(f"Generated {N_samples} samples. Positives percentage: {(y.sum().item() / N_samples) * 100:.2f}%")

    # Split into batches
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    # 2. Build model utilizing our custom Linear module
    model = nn.Sequential(
        CustomLinear(features, 128),
        nn.ReLU(),
        CustomLinear(128, 64),
        nn.ReLU(),
        CustomLinear(64, 1)
    )

    # 3. Proper Kaiming (He) initialization manually applied
    def init_weights(m):
        if isinstance(m, CustomLinear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)
    model.apply(init_weights)

    # 4. Optimizer, Loss & Scheduler
    # We use AdamW instead of Adam to demonstrate decoupled weight decay
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = StableBinaryFocalLoss(alpha=0.8, gamma=2.0) # Highly weight the minority class 1
    
    # Cosine Annealing learning rate schedule
    epochs = 15
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # AMP Setup (CPU fallback or CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model.to(device)

    # Gradient Scaler is used on CUDA for dynamic loss scaling to prevent float16 underflow
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # 5. Training loop
    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        model.train()
        
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            # Forward pass under autocast (mixed precision context)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(batch_X)
                loss = loss_fn(outputs, batch_y)
                
            # Backward pass with scaled loss
            scaler.scale(loss).backward()
            
            # Unscale gradients before clipping to ensure proper scale
            scaler.unscale_(optimizer)
            
            # Gradient clipping to max norm 1.0
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Step the optimizer and the scale factor
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item() * batch_X.size(0)

        scheduler.step()
        avg_loss = epoch_loss / N_samples
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:02d}/{epochs} | Loss: {avg_loss:.6f} | LR: {current_lr:.6f}")

    total_time = time.perf_counter() - t_start
    print(f"Finished Training in {total_time:.3f} seconds.")

    # Inspect our Custom Linear module Buffer state_dict output!
    print(f"Verify state_dict buffer preservation:")
    print(f"  Model custom layer 1 forward pass buffer count: {model[0].forward_passes.item()}")
    print(f"  Keys in model state_dict: {[k for k in model.state_dict().keys() if 'forward_passes' in k]}")


# ============================================================================
# INTERVIEW PREPARATION: QUESTIONS & LIVE CODING EXERCISES
# ============================================================================
"""
A. CONCEPTUAL QUESTIONS

1. EASY: What is the difference between an nn.Parameter and a Buffer in PyTorch?
   - Answer: Both are saved in the model's state_dict, but `nn.Parameter` has 
     `requires_grad=True` by default and is passed to `model.parameters()` (updated 
     by optimizers). Buffers (registered via `register_buffer`) do not track gradients 
     and are used for state metrics like BatchNorm running averages or epoch counters.

2. MEDIUM: What happens if you initialize all weights to exactly 0.0 in a multi-layer perceptron?
   - Answer: This triggers the "symmetry breaking" problem. Because every hidden unit 
     in a layer gets the exact same input weights, they compute identical activation values 
     and receive identical gradients. During backpropagation, all neurons update identically, 
     effectively rendering the width of the hidden layer useless (it acts like a single neuron).

3. HARD: How does the GradScaler in PyTorch AMP handle gradient overflow and underflow?
   - Answer: Underflow: FP16 precision lacks the bits for very small gradients, rounding 
     them to zero. GradScaler multiplies the loss by a large scale factor $S$, scaling 
     gradients into viable FP16 range. 
     Overflow: If the scaled gradients exceed FP16 bounds, they become NaN/Inf. 
     The GradScaler checks gradients; if NaN/Inf are found, it discards the optimizer 
     step entirely, decreases the scale factor $S$ (usually halves it), and repeats. 
     If no overflows occur for a consecutive number of steps, it increases $S$.

B. LIVE CODING EXERCISES

1. EXERCISE: Implement a custom, numerically stable loss function called AsymmetricMSE 
   which penalizes over-predictions (y_hat > y) twice as much as under-predictions (y_hat < y).
   - Code Template to execute:
"""

class AsymmetricMSELoss(nn.Module):
    def __init__(self, over_penalty_weight: float = 2.0):
        super().__init__()
        self.over_penalty_weight = over_penalty_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Errors: e = y_pred - y_true
        errors = y_pred - y_true
        squared_errors = errors ** 2
        
        # Binary mask: 1.0 where prediction is greater than truth
        over_prediction_mask = (errors > 0).float()
        
        # Formulate penalty weight: 2.0 for over-predictions, 1.0 for under-predictions
        weight = over_prediction_mask * self.over_penalty_weight + (1.0 - over_prediction_mask) * 1.0
        
        # Return weighted mean square error
        return torch.mean(weight * squared_errors)


# ============================================================================
# EXECUTING DEMONSTRATIONS
# ============================================================================
if __name__ == "__main__":
    demonstrate_initialization_importance()
    train_focal_loss_model_with_amp()
    
    # Testing the asymmetric loss
    print("\nTesting Custom Asymmetric MSE:")
    loss_fn = AsymmetricMSELoss(over_penalty_weight=2.0)
    y_pred = torch.tensor([1.0, 5.0]) # Error 1: -1.0 (under-prediction), Error 2: +1.0 (over-prediction)
    y_true = torch.tensor([2.0, 4.0])
    # Naive MSE would be: (1^2 + 1^2)/2 = 1.0
    # Asymmetric MSE: (1.0 * 1.0 + 2.0 * 1.0)/2 = 1.5
    print(f"  Calculated Loss: {loss_fn(y_pred, y_true).item()} (Expected: 1.5)")
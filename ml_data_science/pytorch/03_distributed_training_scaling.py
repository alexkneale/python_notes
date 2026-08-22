# ============================================================================
# PYTORCH DEEP DIVE: DISTRIBUTED TRAINING & HETEROGENEOUS SCALING (DDP & FSDP)
# ============================================================================
# Target Audience: Senior ML Engineers / Infrastructure Engineers.
# Topic: GIL Bottle-necks, Data Parallel (DP) vs Distributed Data Parallel (DDP),
#        AllReduce math, ZeRO Memory Sharding (Stage 1-3), and robust DDP boilerplate.
# Compatibility: Python 3.10+, PyTorch 2.0+
# ============================================================================

# ============================================================================
# BEGINNER PRIMER: BASIC CONCEPTS
# ============================================================================
"""
1. What is Distributed Training?
   - When models or datasets become too large for a single machine or GPU, we distribute 
     the workload across multiple devices (GPUs or separate computer nodes).
2. What is Multi-processing?
   - In Python, Multi-threading is limited by the Global Interpreter Lock (GIL), which prevents 
     multiple threads from running Python code at the same time. To leverage multiple GPUs 
     fully, we use Multi-processing: spawning completely separate Python processes for 
     each GPU so they run 100% in parallel.
3. What is Distributed Data Parallel (DDP)?
   - Under DDP, we replicate the exact same model across all GPUs. We split our large dataset 
     into separate portions and feed a different slice to each GPU. Each GPU computes its 
     own gradients; then, they all communicate (using All-Reduce) to sum and average the 
     gradients before updating their weights, keeping all model copies perfectly synchronized.
"""



import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

# To run this script as DDP, you would launch it via torchrun or let mp.spawn handle it.
# This script is executable directly; it will simulate/execute a 2-process DDP spawn on CPU/GPU.

# ============================================================================
# 1. ARCHITECTURAL THEORY: DP VS DDP
# ============================================================================
"""
Why is nn.DataParallel (DP) deprecated/discouraged for high-performance ML?
1. Single Process Multi-Threading: DP runs on a single Python process and leverages 
   multi-threading. Because of Python's Global Interpreter Lock (GIL), only one thread 
   can execute Python bytecode at a time. This severely bottlenecks computation on the CPU side, 
   limiting GPU utilization.
2. Master Node Bottleneck: In DP, GPU 0 acts as the master node. In every forward pass, 
   the input batch is sent to GPU 0, which splits it and scatters sub-batches to other GPUs. 
   GPU 0 also replicates the model on all GPUs. In the backward pass, gradients are gathered 
   back to GPU 0, which updates the weights and scatters the new weights. This results in 
   network congestion and heavy imbalance on GPU 0's memory (leading to Out-Of-Memory errors).

Why is nn.parallel.DistributedDataParallel (DDP) superior?
1. Multi-Process Architecture: DDP spawns a separate Python process for each GPU. 
   There is NO GIL contention.
2. Gradient Ring-AllReduce: DDP replicates the model across all GPUs only ONCE during 
   initialization. In the forward pass, each GPU processes its unique shard of data 
   independently. In the backward pass, gradients are synchronized across all GPUs 
   asynchronously using the high-performance Ring-AllReduce collective operation (through 
   backends like NCCL or Gloo).
3. No Central Master: There is no central master node. Each process maintains its own 
   optimizer and updates its local weights independently. Because they started with the 
   same weights and averaged their gradients, their weights remain perfectly in sync.
"""


# ============================================================================
# 2. ZERO REDUNDANCY OPTIMIZER (ZeRO) & FSDP MECHANICS
# ============================================================================
"""
DDP replicates the model, gradients, and optimizer states across every single GPU.
For large models (e.g. LLMs with billions of parameters), this is extremely wasteful. 
ZeRO (pioneered by DeepSpeed, integrated into PyTorch as FSDP) solves this by sharding memory:

Let Psi be the model size (number of parameters).
For mixed-precision training (FP16/FP32 mixed) with Adam:
  - Model Parameters: 2 bytes (FP16)
  - Gradients: 2 bytes (FP16)
  - Optimizer States (Adam): 12 bytes (FP32 master weight 4B + FP32 momentum 4B + FP32 variance 4B)
  - Total Memory per parameter = 16 bytes!

ZeRO Stages:
- ZeRO-Stage 1: Shards the Optimizer States across GPUs.
  - Memory footprint drops from 16B per GPU to 4B + 12B/N (where N is the number of GPUs).
- ZeRO-Stage 2: Shards both Optimizer States AND Gradients.
  - Memory footprint drops to 2B (parameters) + 14B/N.
- ZeRO-Stage 3: Shards Optimizer States, Gradients, AND Model Parameters.
  - Memory footprint drops to 16B/N. Parameters are fetched via AllGather right before 
    forward/backward computations, and freed immediately after.

Fully Sharded Data Parallel (FSDP) in PyTorch implements ZeRO-Stage 3 directly.
"""


# ============================================================================
# 3. ROBUST DDP BOILERPLATE SETUP (PROCESS GROUP AND PROCESS SPAWNING)
# ============================================================================

def ddp_setup(rank: int, world_size: int):
    """
    Initializes the distributed environment.
    rank: Unique identifier for the current process (0 to world_size - 1).
    world_size: Total number of processes participating in the job.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355" # Any open port
    
    # Backend selection:
    # Use 'nccl' for CUDA (NVIDIA Collective Communications Library - fast, GPU-direct)
    # Use 'gloo' for CPU or testing
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    
    torch.distributed.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def ddp_cleanup():
    """Destroys the distributed environment."""
    torch.distributed.destroy_process_group()


class ToyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)


# ============================================================================
# 4. DISTRIBUTED DATA SAMPLER & THE TRAINING PROCESS
# ============================================================================

def training_process(rank: int, world_size: int):
    """
    This function represents the entire entry-point for a single worker process.
    """
    print(f"[Process {rank}] Starting setup...")
    ddp_setup(rank, world_size)

    # 1. Device mapping
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # 2. Build model and wrap with DDP
    model = ToyNeuralNet().to(device)
    
    # Wrap model with DistributedDataParallel. 
    # For CPU backend testing, we do not specify device_ids.
    if device.type == "cuda":
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    else:
        model = nn.parallel.DistributedDataParallel(model)

    # 3. Create synthetic dataset and Distributed Sampler
    # IMPORTANT: The DistributedSampler restricts data loading to a subset of the dataset 
    # exclusive to the current rank, ensuring workers don't train on duplicate samples!
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = torch.utils.data.TensorDataset(X, y)
    
    sampler = torch.utils.data.DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=True,
        seed=42
    )
    
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=10, 
        pin_memory=True, 
        shuffle=False, # Shuffle must be False when using a distributed sampler!
        sampler=sampler
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print(f"[Process {rank}] Commencing training loop...")
    epochs = 3
    for epoch in range(1, epochs + 1):
        # IMPORTANT: Set sampler epoch to ensure shuffling is different each epoch!
        sampler.set_epoch(epoch)
        
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        # Let's average the loss across all processes using distributed communication
        loss_tensor = torch.tensor(epoch_loss, device=device)
        
        # All-Reduce: Sums the loss_tensor across all processes and stores result in each
        # We then divide by world_size to get the average loss.
        torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / (world_size * len(loader))
        
        if rank == 0:
            print(f"Epoch {epoch:02d} | Synced Average Loss: {avg_loss:.6f}")

    # Synchronize all processes before cleanup
    torch.distributed.barrier()
    ddp_cleanup()
    print(f"[Process {rank}] Finished and cleaned up.")


# ============================================================================
# INTERVIEW PREPARATION: QUESTIONS & LIVE CODING EXERCISES
# ============================================================================
"""
A. CONCEPTUAL QUESTIONS

1. EASY: Why is nn.DataParallel (DP) inefficient compared to DistributedDataParallel (DDP)?
   - Answer: DP uses multi-threading inside a single process, making it subject to 
     Python GIL lock bottlenecks. Additionally, DP uses a master-scatter pattern 
     where GPU 0 collects and redistributes all activations, gradients, and parameters, 
     creating severe network congestion and GPU 0 out-of-memory overheads. DDP spawns 
     one independent process per GPU, avoiding the GIL, and leverages Ring-AllReduce 
     to distribute communications evenly.

2. MEDIUM: Why is calling `sampler.set_epoch(epoch)` inside the DDP epoch loop necessary?
   - Answer: `DistributedSampler` partitions indices based on a random seed. To ensure 
     that training samples are shuffled differently across epochs (and not partitioned 
     the exact same way every epoch), we must pass the current epoch count into the sampler. 
     If omitted, the model will see the exact same batch divisions and sequence order 
     every epoch, lowering training entropy and causing overfitting.

3. HARD: Explain the communication volume and steps of Ring-AllReduce.
   - Answer: Let $N$ be the number of GPUs, and $M$ be the size of the model parameters. 
     In Ring-AllReduce, nodes are arranged in a logical ring. The gradients are divided 
     into $N$ equal chunks. The algorithm has two phases:
     1. Scatter-Reduce: Takes $N-1$ steps. Each step, a node sends a chunk to its neighbor 
        and receives one. At the end, each node holds a fully summed version of one chunk.
     2. All-Gather: Takes $N-1$ steps. Each node broadcasts its fully summed chunk around 
        the ring. At the end, all nodes hold the fully summed gradient array.
     Total data transferred per GPU is $2(N-1) \frac{M}{N}$, which asymptotically approaches 
     $2M$ as $N$ grows. This is highly efficient because bandwidth utilization is fully 
     parallelized and communication cost is independent of $N$.

B. LIVE CODING EXERCISES

1. EXERCISE: Write a helper function `dist_average_tensor(tensor)` that takes a local rank 
   tensor, sums its values across all processes, divides by the world size to get the average, 
   and returns the averaged tensor back to all nodes (simulating loss tracking).
   - Code Template:
"""

def dist_average_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    # 1. Clone tensor to avoid modifying original in-place
    averaged_tensor = tensor.clone()
    
    # 2. Perform All-Reduce using SUM operator
    # If distributed is not initialized (e.g. single-process run), skip
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(averaged_tensor, op=torch.distributed.ReduceOp.SUM)
        # 3. Divide by total processes
        averaged_tensor /= world_size
    return averaged_tensor


# ============================================================================
# MAIN LAUNCHER: USING PYTORCH MULTIPROCESSING
# ============================================================================
if __name__ == "__main__":
    print("="*80)
    print("DISTRIBUTED DATA PARALLEL (DDP) SIMULATION")
    print("="*80)
    
    # We will spawn a 2-process DDP pipeline.
    # Since DDP is multi-processed, we use torch.multiprocessing.spawn
    world_size = 2
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Detected {gpu_count} CUDA GPUs. Spawning DDP with {world_size} processes.")
    else:
        print("CUDA not available. Spawning DDP processes using GLOO backend on CPU.")

    try:
        # mp.spawn spawns 'world_size' processes, calling 'training_process' on each.
        # It automatically injects the 'rank' as the first parameter (0 to world_size-1).
        # Remaining args are passed as a tuple.
        mp.spawn(
            training_process,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
        print("\nDDP Execution finished successfully!")
    except Exception as e:
        print(f"\nExecution terminated. Notes: DDP works best on Linux/macOS with native GPU. Error: {e}")
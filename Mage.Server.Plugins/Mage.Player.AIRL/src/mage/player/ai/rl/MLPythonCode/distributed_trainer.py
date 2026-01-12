#!/usr/bin/env python3
"""
Distributed ML training component for Mage RL system.
Supports PyTorch DistributedDataParallel for multi-GPU training.
"""

import os
import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
import logging
from pathlib import Path
import time
import psutil
import GPUtil
from typing import List, Dict, Any, Optional
import pickle
import gzip
from dataclasses import dataclass
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Metrics collected during training"""
    epoch: int
    loss: float
    episodes_processed: int
    gpu_memory_used: float
    cpu_percent: float
    timestamp: float
    learning_rate: float
    gradient_norm: float


class DistributedRLTrainer:
    """
    Distributed RL trainer with support for multi-GPU and multi-node training.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))

        # Initialize distributed training if available
        if self.world_size > 1:
            self.setup_distributed()

        # Initialize model
        self.model = self.create_model()
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank])

        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=config.get('learning_rate', 1e-4))
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=config.get('lr_step', 1000), gamma=0.9)

        # OpenAI client for embeddings
        self.openai_client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

        # Metrics tracking
        self.metrics_history: List[TrainingMetrics] = []

        # Paths
        self.traj_dir = Path(config.get('traj_dir', '/shared/trajectories'))
        self.model_dir = Path(config.get('model_dir', '/models'))
        self.checkpoint_dir = self.model_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def setup_distributed(self):
        """Initialize distributed training"""
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo')
        torch.cuda.set_device(self.local_rank)
        logger.info(
            f"Distributed training initialized: rank {self.rank}/{self.world_size}")

    def create_model(self):
        """Create the neural network model"""
        # This should match your existing model architecture
        device = torch.device(
            f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu')

        model = torch.nn.Sequential(
            # Adjust input size based on your state representation
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)  # Value function output
        ).to(device)

        return model

    def get_system_metrics(self) -> Dict[str, float]:
        """Collect system metrics"""
        metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'gpu_memory_used': 0.0,
            'gpu_utilization': 0.0
        }

        if torch.cuda.is_available():
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[self.local_rank] if self.local_rank < len(
                        gpus) else gpus[0]
                    metrics['gpu_memory_used'] = gpu.memoryUsed / \
                        gpu.memoryTotal * 100
                    metrics['gpu_utilization'] = gpu.load * 100
            except Exception as e:
                logger.warning(f"Failed to get GPU metrics: {e}")

        return metrics

    def load_trajectory_batch(self, batch_size: int) -> Optional[List[Any]]:
        """Load a batch of trajectory files"""
        trajectory_files = list(self.traj_dir.glob("*.ser.gz"))

        if len(trajectory_files) < batch_size:
            return None

        # Select batch_size files
        selected_files = trajectory_files[:batch_size]
        batch_data = []

        for file_path in selected_files:
            try:
                with gzip.open(file_path, 'rb') as f:
                    episode_data = pickle.load(f)
                    batch_data.append(episode_data)

                # Remove processed file
                file_path.unlink()

            except Exception as e:
                logger.error(f"Failed to load trajectory {file_path}: {e}")
                continue

        return batch_data if batch_data else None

    def train_batch(self, batch_data: List[Any]) -> float:
        """Train on a batch of trajectory data"""
        self.model.train()
        total_loss = 0.0

        for episode_data in batch_data:
            # Extract states, actions, rewards from episode_data
            # This should match your TrainingData structure
            states = self.process_states(episode_data.trajectory)
            returns = torch.tensor(episode_data.returns, dtype=torch.float32)

            if torch.cuda.is_available():
                states = states.cuda(self.local_rank)
                returns = returns.cuda(self.local_rank)

            # Forward pass
            values = self.model(states).squeeze()

            # Compute loss (value function loss)
            loss = torch.nn.functional.mse_loss(values, returns)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(batch_data)

    def process_states(self, trajectory: List[Any]) -> torch.Tensor:
        """Convert trajectory states to tensor format"""
        # This should implement your state representation
        # For now, using placeholder implementation
        state_vectors = []

        for state_data in trajectory:
            # Convert game state to feature vector
            # This should include card embeddings, game state features, etc.
            features = self.extract_features(state_data)
            state_vectors.append(features)

        return torch.stack(state_vectors)

    def extract_features(self, state_data: Any) -> torch.Tensor:
        """Extract features from a single game state"""
        # Placeholder implementation - should match your existing feature extraction
        # This would include card embeddings from OpenAI, game state features, etc.
        return torch.randn(1024)  # Placeholder

    def save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        if self.rank == 0:  # Only save on main process
            checkpoint_path = self.checkpoint_dir / \
                f"checkpoint_epoch_{epoch}.pt"

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'config': self.config,
                'metrics_history': self.metrics_history
            }

            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

    def run_training_loop(self):
        """Main training loop"""
        batch_size = self.config.get('batch_episodes', 50)
        poll_seconds = self.config.get('poll_seconds', 30)
        checkpoint_every = self.config.get('checkpoint_every', 100)

        epoch = 0
        episodes_processed = 0

        logger.info(f"Starting training loop on rank {self.rank}")

        while True:
            # Load batch of trajectories
            batch_data = self.load_trajectory_batch(batch_size)

            if batch_data is None:
                logger.info(
                    f"No trajectories available, waiting {poll_seconds}s...")
                time.sleep(poll_seconds)
                continue

            # Train on batch
            start_time = time.time()
            avg_loss = self.train_batch(batch_data)
            training_time = time.time() - start_time

            # Update scheduler
            self.scheduler.step()

            # Collect metrics
            system_metrics = self.get_system_metrics()
            gradient_norm = self.get_gradient_norm()

            metrics = TrainingMetrics(
                epoch=epoch,
                loss=avg_loss,
                episodes_processed=len(batch_data),
                gpu_memory_used=system_metrics['gpu_memory_used'],
                cpu_percent=system_metrics['cpu_percent'],
                timestamp=time.time(),
                learning_rate=self.scheduler.get_last_lr()[0],
                gradient_norm=gradient_norm
            )

            self.metrics_history.append(metrics)
            episodes_processed += len(batch_data)

            # Log progress
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}: Loss={avg_loss:.4f}, "
                    f"Episodes={episodes_processed}, "
                    f"LR={metrics.learning_rate:.6f}, "
                    f"GPU={metrics.gpu_memory_used:.1f}%, "
                    f"Time={training_time:.2f}s"
                )

            # Save checkpoint
            if epoch % checkpoint_every == 0:
                self.save_checkpoint(epoch)

            epoch += 1

    def get_gradient_norm(self) -> float:
        """Calculate gradient norm for monitoring"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def cleanup(self):
        """Cleanup distributed training"""
        if self.world_size > 1:
            dist.destroy_process_group()


def main():
    """Main entry point for distributed training"""
    config = {
        'learning_rate': float(os.environ.get('LEARNING_RATE', 1e-4)),
        'batch_episodes': int(os.environ.get('BATCH_EPISODES', 50)),
        'poll_seconds': int(os.environ.get('POLL_SECONDS', 30)),
        'checkpoint_every': int(os.environ.get('CHECKPOINT_EVERY', 100)),
        'traj_dir': os.environ.get('TRAJ_DIR', '/shared/trajectories'),
        'model_dir': os.environ.get('MODEL_DIR', '/models'),
    }

    trainer = DistributedRLTrainer(config)

    try:
        trainer.run_training_loop()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
    finally:
        trainer.cleanup()


if __name__ == '__main__':
    main()

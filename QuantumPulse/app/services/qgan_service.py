import logging
import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# --- QGAN Configuration ---
N_QUBITS = 4  # Number of qubits for the generator
N_LAYERS = 3  # Number of layers in the quantum circuit
LATENT_DIM = 4 # Dimension of the latent space for the generator
DEVICE = "default.qubit" # PennyLane device to use for simulation
BATCH_SIZE = 32
LEARNING_RATE_G = 0.01
LEARNING_RATE_D = 0.001

# --- Quantum Generator ---
dev = qml.device(DEVICE, wires=N_QUBITS)

def quantum_generator_circuit(noise, weights):
    """
    Quantum circuit for the QGAN generator.
    Uses parameterized quantum circuit (PQC) architecture.
    """
    # Input encoding layer
    for i in range(N_QUBITS):
        qml.RY(noise[i % len(noise)] * np.pi, wires=i)
    
    # Variational layers
    for layer in range(N_LAYERS):
        # Rotation layer
        for i in range(N_QUBITS):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
        
        # Entanglement layer
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
        if N_QUBITS > 2:
            qml.CNOT(wires=[N_QUBITS - 1, 0])  # Circular entanglement
    
    # Final rotation layer
    for i in range(N_QUBITS):
        qml.RY(weights[-1, i, 0], wires=i)
    
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

# Create quantum node with gradient support
qnode = qml.QNode(quantum_generator_circuit, dev, interface="torch", diff_method="parameter-shift")

# --- Classical Discriminator ---
class Discriminator(nn.Module):
    """Enhanced classical discriminator for the QGAN."""
    def __init__(self, input_dim: int = N_QUBITS):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)

    def forward(self, x):
        return self.model(x)

# --- QGAN Service ---
class QGANService:
    """
    Manages the lifecycle of a Quantum Generative Adversarial Network.
    Implements proper adversarial training between quantum generator and classical discriminator.
    """
    def __init__(self):
        # Initialize quantum generator parameters
        self.generator_weights = nn.Parameter(
            torch.randn(N_LAYERS + 1, N_QUBITS, 2) * 0.1, 
            requires_grad=True
        )
        
        # Initialize discriminator
        self.discriminator = Discriminator()
        
        # Optimizers
        self.g_optimizer = torch.optim.Adam([self.generator_weights], lr=LEARNING_RATE_G)
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=LEARNING_RATE_D)
        
        # Loss function
        self.loss_fn = nn.BCELoss()
        
        # Training state
        self.is_trained = False
        self.training_history = {
            'g_losses': [],
            'd_losses': [],
            'd_accuracies': [],
            'epochs': 0
        }
        
        # Model metadata
        self.model_id = f"qgan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"QGANService initialized with model_id: {self.model_id}")

    def _generate_batch(self, batch_size: int) -> torch.Tensor:
        """Generate a batch of samples using the quantum generator."""
        samples = []
        
        for _ in range(batch_size):
            # Generate random noise
            noise = torch.randn(LATENT_DIM)
            
            # Pass through quantum circuit
            output = qnode(noise, self.generator_weights)
            samples.append(output)
        
        return torch.stack(samples)
    
    def _calculate_gradient_penalty(self, real_data: torch.Tensor, fake_data: torch.Tensor, lambda_gp: float = 10.0) -> torch.Tensor:
        """Calculate gradient penalty for Wasserstein GAN training (optional enhancement)."""
        batch_size = real_data.size(0)
        epsilon = torch.rand(batch_size, 1)
        epsilon = epsilon.expand_as(real_data)
        
        interpolated = epsilon * real_data + (1 - epsilon) * fake_data
        interpolated.requires_grad_(True)
        
        prob_interpolated = self.discriminator(interpolated)
        
        gradients = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(prob_interpolated),
            create_graph=True,
            retain_graph=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return lambda_gp * gradient_penalty

    def train(self, real_data: List[List[float]], epochs: int = 50, batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Trains the QGAN on a set of real data using proper adversarial training.
        
        Args:
            real_data: Training data samples
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary containing training results and metrics
        """
        logger.info(f"Starting QGAN training for {epochs} epochs with {len(real_data)} samples.")
        
        # Convert data to tensor
        real_data_tensor = torch.FloatTensor(real_data)
        data_size = len(real_data_tensor)
        batch_size = batch_size or min(BATCH_SIZE, data_size)
        
        # Training metrics
        g_losses = []
        d_losses = []
        d_accuracies = []
        
        for epoch in range(epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_d_acc = 0.0
            n_batches = 0
            
            # Shuffle data
            indices = torch.randperm(data_size)
            
            for i in range(0, data_size, batch_size):
                batch_indices = indices[i:i + batch_size]
                real_batch = real_data_tensor[batch_indices]
                current_batch_size = len(real_batch)
                
                # =============================
                # Train Discriminator
                # =============================
                self.d_optimizer.zero_grad()
                
                # Real data
                real_labels = torch.ones(current_batch_size, 1) * 0.9  # Label smoothing
                real_pred = self.discriminator(real_batch)
                d_loss_real = self.loss_fn(real_pred, real_labels)
                
                # Fake data
                fake_batch = self._generate_batch(current_batch_size)
                fake_labels = torch.zeros(current_batch_size, 1) + 0.1  # Label smoothing
                fake_pred = self.discriminator(fake_batch.detach())
                d_loss_fake = self.loss_fn(fake_pred, fake_labels)
                
                # Total discriminator loss
                d_loss = (d_loss_real + d_loss_fake) / 2
                d_loss.backward()
                
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                
                self.d_optimizer.step()
                
                # Calculate discriminator accuracy
                real_acc = (real_pred > 0.5).float().mean()
                fake_acc = (fake_pred < 0.5).float().mean()
                d_acc = (real_acc + fake_acc) / 2
                
                # =============================
                # Train Generator
                # =============================
                self.g_optimizer.zero_grad()
                
                # Generate new fake batch
                fake_batch = self._generate_batch(current_batch_size)
                fake_labels = torch.ones(current_batch_size, 1)  # Generator wants to fool discriminator
                fake_pred = self.discriminator(fake_batch)
                g_loss = self.loss_fn(fake_pred, fake_labels)
                
                g_loss.backward()
                
                # Clip gradients for quantum parameters
                torch.nn.utils.clip_grad_norm_([self.generator_weights], max_norm=1.0)
                
                self.g_optimizer.step()
                
                # Accumulate metrics
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                epoch_d_acc += d_acc.item()
                n_batches += 1
            
            # Average metrics for epoch
            avg_g_loss = epoch_g_loss / n_batches
            avg_d_loss = epoch_d_loss / n_batches
            avg_d_acc = epoch_d_acc / n_batches
            
            g_losses.append(avg_g_loss)
            d_losses.append(avg_d_loss)
            d_accuracies.append(avg_d_acc)
            
            # Log progress
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}/{epochs} - "
                    f"G Loss: {avg_g_loss:.4f}, "
                    f"D Loss: {avg_d_loss:.4f}, "
                    f"D Accuracy: {avg_d_acc:.4f}"
                )
        
        # Update training state
        self.is_trained = True
        self.training_history['g_losses'].extend(g_losses)
        self.training_history['d_losses'].extend(d_losses)
        self.training_history['d_accuracies'].extend(d_accuracies)
        self.training_history['epochs'] += epochs
        
        # Calculate final metrics
        final_g_loss = np.mean(g_losses[-10:]) if len(g_losses) >= 10 else np.mean(g_losses)
        final_d_loss = np.mean(d_losses[-10:]) if len(d_losses) >= 10 else np.mean(d_losses)
        final_d_acc = np.mean(d_accuracies[-10:]) if len(d_accuracies) >= 10 else np.mean(d_accuracies)
        
        logger.info(
            f"QGAN training complete. "
            f"Final G Loss: {final_g_loss:.4f}, "
            f"Final D Loss: {final_d_loss:.4f}, "
            f"Final D Accuracy: {final_d_acc:.4f}"
        )
        
        return {
            "status": "success",
            "model_id": self.model_id,
            "epochs": epochs,
            "final_g_loss": float(final_g_loss),
            "final_d_loss": float(final_d_loss),
            "final_d_accuracy": float(final_d_acc),
            "training_time": datetime.now().isoformat(),
            "data_samples": len(real_data),
            "quantum_params": {
                "n_qubits": N_QUBITS,
                "n_layers": N_LAYERS,
                "latent_dim": LATENT_DIM
            }
        }

    def generate_samples(self, n_samples: int, return_metadata: bool = False) -> Union[List[List[float]], Tuple[List[List[float]], Dict[str, Any]]]:
        """
        Generates new data samples using the trained quantum generator.
        
        Args:
            n_samples: Number of samples to generate
            return_metadata: If True, return additional metadata about generation
            
        Returns:
            Generated samples (and optionally metadata)
        """
        if not self.is_trained:
            logger.warning("Generator is not trained. Training with random data first.")
            # Generate some random training data
            random_data = torch.randn(100, N_QUBITS).tolist()
            self.train(random_data, epochs=20)
        
        logger.info(f"Generating {n_samples} new samples using quantum generator.")
        
        generated_samples = []
        generation_times = []
        
        with torch.no_grad():
            for i in range(n_samples):
                start_time = datetime.now()
                
                # Generate random noise
                noise = torch.randn(LATENT_DIM)
                
                # Pass through quantum circuit
                output = qnode(noise, self.generator_weights)
                generated_samples.append(output.tolist())
                
                generation_times.append((datetime.now() - start_time).total_seconds())
                
                if (i + 1) % 10 == 0:
                    logger.debug(f"Generated {i + 1}/{n_samples} samples")
        
        logger.info(f"Successfully generated {n_samples} samples.")
        
        if return_metadata:
            metadata = {
                "model_id": self.model_id,
                "n_samples": n_samples,
                "generation_time_seconds": sum(generation_times),
                "avg_time_per_sample": np.mean(generation_times),
                "is_trained": self.is_trained,
                "training_epochs": self.training_history['epochs'],
                "timestamp": datetime.now().isoformat()
            }
            return generated_samples, metadata
        
        return generated_samples
    
    def evaluate_quality(self, generated_samples: List[List[float]], real_samples: List[List[float]]) -> Dict[str, float]:
        """
        Evaluate the quality of generated samples compared to real samples.
        
        Args:
            generated_samples: Samples generated by the QGAN
            real_samples: Real data samples for comparison
            
        Returns:
            Dictionary containing quality metrics
        """
        gen_tensor = torch.FloatTensor(generated_samples)
        real_tensor = torch.FloatTensor(real_samples)
        
        # Calculate discriminator scores
        with torch.no_grad():
            gen_scores = self.discriminator(gen_tensor).numpy()
            real_scores = self.discriminator(real_tensor).numpy()
        
        # Calculate statistical metrics
        gen_mean = np.mean(generated_samples, axis=0)
        real_mean = np.mean(real_samples, axis=0)
        gen_std = np.std(generated_samples, axis=0)
        real_std = np.std(real_samples, axis=0)
        
        # Calculate distances
        mean_distance = np.linalg.norm(gen_mean - real_mean)
        std_distance = np.linalg.norm(gen_std - real_std)
        
        # Discriminator-based metrics
        avg_gen_score = float(np.mean(gen_scores))
        avg_real_score = float(np.mean(real_scores))
        
        return {
            "mean_distance": float(mean_distance),
            "std_distance": float(std_distance),
            "avg_generator_score": avg_gen_score,
            "avg_real_score": avg_real_score,
            "discriminator_gap": float(avg_real_score - avg_gen_score),
            "generator_quality": float(avg_gen_score / avg_real_score) if avg_real_score > 0 else 0.0
        }
    
    def save_model(self, path: str) -> None:
        """Save the QGAN model state."""
        state = {
            'model_id': self.model_id,
            'generator_weights': self.generator_weights.detach().cpu().numpy().tolist(),
            'discriminator_state': self.discriminator.state_dict(),
            'training_history': self.training_history,
            'is_trained': self.is_trained,
            'config': {
                'n_qubits': N_QUBITS,
                'n_layers': N_LAYERS,
                'latent_dim': LATENT_DIM
            }
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load a saved QGAN model state."""
        with open(path, 'r') as f:
            state = json.load(f)
        
        self.model_id = state['model_id']
        self.generator_weights = nn.Parameter(torch.tensor(state['generator_weights'], dtype=torch.float32))
        self.discriminator.load_state_dict(state['discriminator_state'])
        self.training_history = state['training_history']
        self.is_trained = state['is_trained']
        
        # Reinitialize optimizers with loaded parameters
        self.g_optimizer = torch.optim.Adam([self.generator_weights], lr=LEARNING_RATE_G)
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=LEARNING_RATE_D)
        
        logger.info(f"Model loaded from {path}")

# --- Singleton Instance ---
qgan_service = QGANService() 
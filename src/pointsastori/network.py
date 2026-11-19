import time
import pickle
import numpy as np

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx

from dataclasses import dataclass

from .linear_least_squares import least_squares_coefficients
from .torus_utils import orthonormal_basis_with_rotation
from .shape_3d import *


NAN_EPSILON = 1e-10

# ========================================================================
# NEURAL NETWORK ARCHITECTURE
# ========================================================================

@dataclass
class ModelConfig:
    n_layers: int
    mha_width: int
    mlp_width: int
    n_heads: int


class TransformerBlock(nnx.Module):
    """
    Transformer block using Flax built-in modules.
    """

    def __init__(self, embed_dim: int, n_heads: int, mlp_dim: int, dropout_rate: float = 0.0, *, rngs: nnx.Rngs):
        self.attention = nnx.MultiHeadAttention(
            num_heads=n_heads,
            in_features=embed_dim,
            qkv_features=embed_dim,
            out_features=embed_dim,
            decode=False,
            rngs=rngs,
        )

        # feed-forward network
        self.ffn = nnx.Sequential(
            nnx.Linear(embed_dim, mlp_dim, rngs=rngs),
            nnx.relu,
            nnx.Dropout(rate=dropout_rate, rngs=rngs),
            nnx.Linear(mlp_dim, embed_dim, rngs=rngs),
            nnx.Dropout(rate=dropout_rate, rngs=rngs),
        )

        # layer normalization
        self.norm1 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # Self-attention block
        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = residual + x

        # Feed-forward block
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x


class InputFeatures(nnx.Module):
    """
    Turn a neighborhood (point positions & their normals) to the input features to be fed into the neural network.
    """

    def __init__(self, output_dim: int, n_freq: int = 0, *, rngs: nnx.Rngs):
        # Input features: local coordinates (3), normals in local coordinates (3)
        input_dim = 6

        self.n_freq = n_freq
        if n_freq > 0:
            pos_encoding_dim = 3 * (2 * n_freq + 1)
            input_dim = pos_encoding_dim + 3

        # A linear projection simply to match transformer dimension
        self.mlp = nnx.Sequential(
            nnx.Linear(input_dim, output_dim, rngs=rngs),
            #nnx.relu,
            #nnx.Linear(2*output_dim, output_dim, rngs=rngs),
            nnx.LayerNorm(output_dim, rngs=rngs),  # LayerNorm seems to really help here
        )

        self.key = jax.random.PRNGKey(57)

    def __call__(
        self,
        positions: jax.Array,  # [batch, k, 3]
        normals: jax.Array,  # [batch, k, 3]
    ) -> jax.Array:
        """
        Compute geometric features relative to first point (center).

        Returns:
            [batch, k, output_dim] geometric encodings
        """
        batch_size, k, _ = positions.shape

        norms = jnp.linalg.norm(normals, axis=-1, keepdims=True)  # [batch, k, 1]
        normalized_normals = normals / jnp.maximum(norms, NAN_EPSILON)  # [batch, k, 3]

        # Center point (index 0)
        center_normal = normalized_normals[:, 0:1, :]  # [batch, 1, 3]

        # Regularization: scale neighborhood according to median_j |p_i - p_j|
        diff_vectors = positions - positions[:, 0:1, :]  # [batch, k, 3]
        distances = jnp.sqrt(jnp.sum(diff_vectors * diff_vectors, axis=-1))  # [batch, k]
        median = jnp.median(distances, axis=-1)  # [batch, ]
        normalized_positions = positions / median[:, None, None]  # [batch, k, 3]

        # The following block of code is copied from local_coordinates() above, because we also need the orthonormal basis.
        center_point = normalized_positions[:, 0:1, :]  # [batch, 1, 3]
        s_hat, t_hat = orthonormal_basis_with_rotation(center_normal, 0.0)

        # Encode relative positions in local frame as (s, t, Q(s, t)) tuples.
        diff_vectors = normalized_positions - center_point  # [batch, k, 3]
        heights = jnp.sum(diff_vectors * center_normal, axis=-1, keepdims=True)  # [batch, k, 1]
        S = jnp.sum(diff_vectors * s_hat, axis=-1, keepdims=True)  # [batch, k, 1]
        T = jnp.sum(diff_vectors * t_hat, axis=-1, keepdims=True)  # [batch, k, 1]
        coordinates = jnp.concatenate([S, T, heights], axis=-1)  # [batch, k, 3]

        # Encode normals in local frame.
        S = jnp.sum(normalized_normals * s_hat, axis=-1, keepdims=True)  # [batch, k, 1]
        T = jnp.sum(normalized_normals * t_hat, axis=-1, keepdims=True)  # [batch, k, 1]
        N = jnp.sum(normalized_normals * center_normal, axis=-1, keepdims=True)  # [batch, k, 1]
        local_normals = jnp.concatenate([S, T, N], axis=-1)  # [batch, k, 3]

        # [local coordinates (3), normals in local coordinates (3)]
        geometric_features = jnp.concatenate([coordinates, local_normals], axis=-1)  # [batch, k, input_dim]

        return self.mlp(geometric_features), median


class InputFeaturesWithUncertainty(nnx.Module):
    """
    Turn a neighborhood (point positions & their normals) to the input features to be fed into the neural network.
    """

    def __init__(self, mlp_dim: int, output_dim: int, n_freq: int = 0, *, rngs: nnx.Rngs):
        # Input features: local coordinates (3), normals in local coordinates (3)
        input_dim = 6

        self.n_freq = n_freq
        if n_freq > 0:
            pos_encoding_dim = 3 * (2 * n_freq + 1)
            input_dim = pos_encoding_dim + 3

        # A linear projection simply to match transformer dimension
        self.mlp = nnx.Sequential(
            nnx.Linear(input_dim, mlp_dim, rngs=rngs),
            nnx.LayerNorm(mlp_dim, rngs=rngs),  # LayerNorm seems to really help here
            nnx.relu,
            nnx.Linear(mlp_dim, output_dim, rngs=rngs),
            nnx.LayerNorm(output_dim, rngs=rngs),  # LayerNorm seems to really help here
        )

        self.key = jax.random.PRNGKey(57)

    def __call__(
        self,
        positions: jax.Array,  # [batch, k, 3]
        normals: jax.Array,  # [batch, k, 3]
    ) -> jax.Array:
        """
        Compute geometric features relative to first point (center).

        Returns:
            [batch, k, output_dim] geometric encodings
        """
        n_devices, batch_size, k, _ = positions.shape

        norms = jnp.linalg.norm(normals, axis=-1, keepdims=True)  # [device, batch, k, 1]
        normalized_normals = normals / jnp.maximum(norms, NAN_EPSILON)  # [device, batch, k, 3]

        # Center point (index 0)
        center_normal = normalized_normals[:, :, 0:1, :]  # [device, batch, 1, 3]

        # Regularization: scale neighborhood according to median_j |p_i - p_j|
        diff_vectors = positions - positions[:, :, 0:1, :]  # [device, batch, k, 3]
        distances = jnp.sqrt(jnp.sum(diff_vectors * diff_vectors, axis=-1))  # [batch, k]
        median = jnp.median(distances, axis=-1)  # [batch, ]
        normalized_positions = positions / median[:, :, None, None]  # [device, batch, k, 3]

        # The following block of code is copied from local_coordinates() above, because we also need the orthonormal basis.
        center_point = normalized_positions[:, :, 0:1, :]  # [device, batch, 1, 3]
        s_hat, t_hat = orthonormal_basis_with_rotation(center_normal, 0.0)

        # Encode relative positions in local frame as (s, t, Q(s, t)) tuples.
        diff_vectors = normalized_positions - center_point  # [device, batch, k, 3]
        heights = jnp.sum(diff_vectors * center_normal, axis=-1, keepdims=True)  # [device, batch, k, 1]
        S = jnp.sum(diff_vectors * s_hat, axis=-1, keepdims=True)  # [device, batch, k, 1]
        T = jnp.sum(diff_vectors * t_hat, axis=-1, keepdims=True)  # [device, batch, k, 1]
        coordinates = jnp.concatenate([S, T, heights], axis=-1)  # [device, batch, k, 3]

        # Encode normals in local frame.
        S = jnp.sum(normalized_normals * s_hat, axis=-1, keepdims=True)  # [device, batch, k, 1]
        T = jnp.sum(normalized_normals * t_hat, axis=-1, keepdims=True)  # [device, batch, k, 1]
        N = jnp.sum(normalized_normals * center_normal, axis=-1, keepdims=True)  # [device, batch, k, 1]
        local_normals = jnp.concatenate([S, T, N], axis=-1)  # [device, batch, k, 3]

        # [local coordinates (3), normals in local coordinates (3)]
        geometric_features = jnp.concatenate([coordinates, local_normals], axis=-1)  # [device, batch, k, input_dim]

        return self.mlp(geometric_features), median


class FundamentalFormPredictor(nnx.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        n_freq: int = 0,
        n_layers: int = 4,
        n_heads: int = 4,
        mlp_dim: int = 256,
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Args: Parameters for each TransformerBlock.
            n_layers: Number of TransformerBlocks.
        """

        self.input_encoder = InputFeatures(embed_dim, n_freq=n_freq, rngs=rngs)

        self.blocks = nnx.List(
            [TransformerBlock(embed_dim, n_heads, mlp_dim, dropout_rate, rngs=rngs) for _ in range(n_layers)]
        )

        self.final_norm = nnx.LayerNorm(embed_dim, rngs=rngs)

        # Projection
        # (a00, a01, a10, a11, a02, a20)
        self.mlp = nnx.Sequential(
            nnx.Linear(
                embed_dim,
                6,
                kernel_init=nnx.initializers.zeros_init(),
                # bias_init=nnx.initializers.ones_init(),
                bias_init=nnx.initializers.constant(jnp.array([0.0, 0.0, 0.0, 0.0, -0.5, -0.5])),
                rngs=rngs,
            ),
        )

    @nnx.jit
    def __call__(self, positions: jax.Array, normals: jax.Array) -> jax.Array:
        """
        Args:
            positions, normals: [batch, k, 3] representing a neighborhood of size k; the first point (index 0) is the center

        Returns:
            coefficients: [batch, 6] (a00, a01, a10, a11, a02, a20)
        """
        batch_size, k, d = positions.shape

        # input_features: [batch, k, embed_dim]
        # scale: [batch, ]
        input_features, scale = self.input_encoder(positions, normals)
        x = input_features

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)  # [batch, k, embed_dim]

        # Extract center point
        x = x[:, 0, :]  # [batch, embed_dim]

        # Predict form coefficients
        coeffs = self.mlp(x)  # [batch, 6]

        # Scale back to original space
        coeffs = jnp.stack(
            [
                coeffs[:, 0] * scale,  # a00
                coeffs[:, 1],  # a01
                coeffs[:, 2],  # a10
                coeffs[:, 3] / scale,  # a11
                coeffs[:, 4] / scale,  # a02
                coeffs[:, 5] / scale,  # a20
            ],
            axis=-1,
        )

        return coeffs

    def precompute_coefficients(self, points, normals, k_neighbors, outliers=np.array([])):
        nnx.eval_mode(self)
        # Generate neighborhoods
        indices = get_neighbors(points, k_neighbors, outliers)  # (|P|, k)

        # Run forward evaluation.
        predicted_coefficients = self(points[indices], normals[indices])  # (|P|, 6)

        return predicted_coefficients

    def precompute_coefficients_in_chunks(self, points, normals, k_neighbors, chunk_size=50000, outliers=np.array([]), use_linear_least_squares: bool = False):
        """
        Process in chunks to avoid GPU memory issues
        """
        nnx.eval_mode(self)

        n_points = points.shape[0]
        print(f'Processing {n_points} points in chunks of {chunk_size}...')

        # Generate patches
        t0 = time.time()
        indices = get_neighbors(points, k_neighbors, outliers)  # (|P|, k)
        t1 = time.time()
        print(f'  get_neighbors: {t1 - t0:.2f}s')

        # Allocate output array
        all_coefficients = []

        # Process in chunks
        n_chunks = (n_points + chunk_size - 1) // chunk_size
        t0 = time.time()

        if use_linear_least_squares:
            jit_least_squares = jax.jit(least_squares_coefficients)

        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, n_points)

            # Extract chunk
            chunk_indices = indices[start_idx:end_idx]  # (chunk_size, k)
            chunk_points = points[chunk_indices]  # (chunk_size, k, 3)
            chunk_normals = normals[chunk_indices]  # (chunk_size, k, 3)

            if use_linear_least_squares:
                chunk_coeffs = jit_least_squares(chunk_points, chunk_normals)

            else:
                # Forward pass for this chunk
                chunk_coeffs = self(chunk_points, chunk_normals)

            # Wait for GPU to finish and transfer to CPU
            chunk_coeffs_cpu = np.array(chunk_coeffs.block_until_ready())
            all_coefficients.append(chunk_coeffs_cpu)

            # Clear JAX cache for this chunk
            del chunk_coeffs

            if (chunk_idx + 1) % 10 == 0:
                print(f'    {chunk_idx + 1}/{n_chunks} chunks processed')

        t1 = time.time()
        print(f'  Coefficient prediction (chunked): {t1 - t0:.2f}s')

        # Concatenate all chunks
        predicted_coefficients = np.concatenate(all_coefficients, axis=0)

        return predicted_coefficients

    @staticmethod
    def load_saved_model(filepath: str):
        """
        Given the filepath where weights are saved, return the model and its parameters.
        """
        with open(filepath, 'rb') as f:
            saved_data = pickle.load(f)

        n_freq = saved_data['n_freq'] if 'n_freq' in saved_data else 0

        loaded_model = FundamentalFormPredictor(
            embed_dim=saved_data['embed_dim'],
            n_freq=n_freq,
            n_heads=saved_data['n_heads'],
            n_layers=saved_data['n_layers'],
            mlp_dim=saved_data['mlp_dim'],
            dropout_rate=saved_data['dropout_rate'],  # not used in eval mode
            rngs=nnx.Rngs(0),  # not used in eval mode
        )

        # Load saved parameters into the model
        nnx.update(loaded_model, saved_data['params'])

        # Put in eval mode
        nnx.eval_mode(loaded_model)

        print(f'Loaded FundamentalFormPredictor model:')
        print(f'  embed_dim: {saved_data["embed_dim"]}')
        print(f'  n_freq: {n_freq}')
        print(f'  n_heads: {saved_data["n_heads"]}')
        print(f'  n_layers: {saved_data["n_layers"]}')
        print(f'  mlp_dim: {saved_data["mlp_dim"]}')
        print(f'  dropout_rate: {saved_data["dropout_rate"]}')

        return loaded_model, saved_data['k_neighbors']


class FundamentalFormPredictorWithUncertainty(nnx.Module):
    def __init__(
        self,
        config: ModelConfig,
        rngs: nnx.Rngs,
        dropout_rate: float = 0.0,
    ):
        """
        Args: Parameters for each TransformerBlock.
            n_layers: Number of TransformerBlocks.
        """

        self.n_layers = config.n_layers
        self.mha_width = config.mha_width
        self.mlp_width = config.mlp_width
        self.n_heads = config.n_heads

        self.checkpointer = ocp.StandardCheckpointer()

        self.input_encoder = InputFeaturesWithUncertainty(
            config.mlp_width,
            config.mha_width,
            n_freq=0,
            rngs=rngs
        )

        self.blocks = nnx.List(
            [TransformerBlock(
                config.mha_width,
                config.n_heads,
                config.mlp_width,
                dropout_rate,
                rngs=rngs)
             for _ in range(config.n_layers)]
        )

        self.final_norm = nnx.LayerNorm(config.mha_width, rngs=rngs)

        # Projection
        # (a00, a01, a10, a11, a02, a20, uncertainty)
        self.mlp = nnx.Sequential(
            nnx.Linear(
                config.mha_width,
                7,
                kernel_init=nnx.initializers.zeros_init(),
                bias_init=nnx.initializers.constant(
                    jnp.array([0.0, 0.0, 0.0, 0.0, -0.5, -0.5, 0.75]
                )),
                rngs=rngs,
            ),
        )

    @nnx.jit
    def __call__(self, positions: jax.Array, normals: jax.Array) -> jax.Array:
        """
        Args:
            positions, normals: [batch, k, 3] representing a neighborhood of size k; the first point (index 0) is the center

        Returns:
            coefficients: [batch, 6] (a00, a01, a10, a11, a02, a20)
        """
        n_shapes, batch_size, k, d = positions.shape

        # input_features: [batch, k, embed_dim]
        # scale: [batch, ]
        input_features, scale = self.input_encoder(positions, normals)
        x = input_features

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)  # [batch, k, embed_dim]

        # Extract center point
        x = x[:, :, 0, :]  # [device, batch, embed_dim]

        # Predict form coefficients
        coeffs = self.mlp(x)  # [device, batch, 10]

        # Scale back to original space
        coeffs = jnp.stack(
            [
                coeffs[:, :, 0] * scale,  # a00
                coeffs[:, :, 1],  # a01
                coeffs[:, :, 2],  # a10
                coeffs[:, :, 3] / scale,  # a11
                coeffs[:, :, 4] / scale,  # a02
                coeffs[:, :, 5] / scale,  # a20
                coeffs[:, :, 6] # uncertainty
            ],
            axis=-1,
        )

        return coeffs

    def precompute_coefficients(self, points, normals, k_neighbors, outliers=np.array([])):
        self.eval()
        # Generate neighborhoods
        indices = get_neighbors(points, k_neighbors, outliers)  # (|P|, k)

        # Run forward evaluation.
        predicted_coefficients = self(points[indices], normals[indices])  # (|P|, 7)

        return predicted_coefficients

    def precompute_coefficients_in_chunks(
        self,
        points,
        normals,
        k_neighbors,
        chunk_size=50000,
        outliers=np.array([]),
        use_linear_least_squares: bool = False
    ):
        """
        Process in chunks to avoid GPU memory issues
        """
        self.eval()

        n_points = points.shape[0]
        print(f'Processing {n_points} points in chunks of {chunk_size}...')

        # Generate patches
        t0 = time.time()
        indices = get_neighbors(points, k_neighbors, outliers)  # (|P|, k)
        t1 = time.time()
        print(f'  get_neighbors: {t1 - t0:.2f}s')

        # Allocate output array
        all_coefficients = []

        # Process in chunks
        n_chunks = (n_points + chunk_size - 1) // chunk_size
        t0 = time.time()

        if use_linear_least_squares:
            jit_least_squares = jax.jit(least_squares_coefficients)

        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, n_points)

            # Extract chunk
            chunk_indices = indices[start_idx:end_idx]  # (chunk_size, k)
            chunk_points = points[chunk_indices]  # (chunk_size, k, 3)
            chunk_normals = normals[chunk_indices]  # (chunk_size, k, 3)

            if use_linear_least_squares:
                chunk_coeffs = jit_least_squares(chunk_points, chunk_normals)

            else:
                # Forward pass for this chunk
                chunk_coeffs = self(chunk_points, chunk_normals)

            # Wait for GPU to finish and transfer to CPU
            chunk_coeffs_cpu = np.array(chunk_coeffs.block_until_ready())
            all_coefficients.append(chunk_coeffs_cpu)

            # Clear JAX cache for this chunk
            del chunk_coeffs

            if (chunk_idx + 1) % 10 == 0:
                print(f'    {chunk_idx + 1}/{n_chunks} chunks processed')

        t1 = time.time()
        print(f'  Coefficient prediction (chunked): {t1 - t0:.2f}s')

        # Concatenate all chunks
        predicted_coefficients = np.concatenate(all_coefficients, axis=0)

        return predicted_coefficients

    def restore_from_checkpoint(self, path: str):
        """Restore model state from orbax checkpoint

        Args:
            path (str): Absolute path to the intended checkpoint
        """
        graphdef, abstract_state = nnx.split(self)
        state_restored = self.checkpointer.restore(path, abstract_state)
        return nnx.merge(graphdef, state_restored)

    def save_to_checkpoint(self, path: str):
        """Save model state to orbax checkpoint

        Args:
            path (str): Absolute path to the intended checkpoint
        """
        _, state = nnx.split(self)
        self.checkpointer.save(path, state)
        self.checkpointer.wait_until_finished()

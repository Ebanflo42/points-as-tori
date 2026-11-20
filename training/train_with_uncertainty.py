import os
import glob
import time
import yaml
import numpy as np

from dataclasses import dataclass
from argparse import ArgumentParser

import jax
import optax
import jax.numpy as jnp

from flax import nnx
from jax.sharding import PartitionSpec, NamedSharding
from jax.experimental import mesh_utils

from pointsastori.mesh_sampling_jax import generate_shape_features
from pointsastori.network import FundamentalFormPredictorWithUncertainty, ModelConfig
from pointsastori.torus_utils import torus_precompute_masked_sharded, torus_distance_gradient_masked_with_uncertainty
from train import load_mesh_training_data

from typing import *


@dataclass
class TrainingConfig:
    learning_rate: float
    warmup_steps: int
    num_epochs: int
    queries_per_neighborhood: int
    save_and_validate_every: int
    print_every: int


def build_device_mesh():
    num_devices = jax.local_device_count()
    assert num_devices >= 1
    mesh = jax.sharding.Mesh(
        mesh_utils.create_device_mesh((num_devices,)),
        ('data',),
    )
    model_sharding = NamedSharding(mesh, PartitionSpec())       # replicated
    data_sharding  = NamedSharding(mesh, PartitionSpec('data')) # sharded on axis 0
    return mesh, model_sharding, data_sharding


def SDF_loss_with_uncertainty(
    lambda_coefficient,
    uncertainty_coefficient,
    mask,
    positions,
    normals,
    coefficients,
    uncertainties,
    queries,
    true_distances
):
    uncertainties_pos = nnx.softplus(uncertainties)

    centers, axes, major_radii, minor_radii = torus_precompute_masked_sharded(
        mask, positions, normals, coefficients
    )
    predicted_distances, predicted_gradients, interpolated_uncertainty = \
        torus_distance_gradient_masked_with_uncertainty(
        lambda_coefficient,
        mask,
        queries,
        positions,
        centers,
        axes,
        major_radii,
        minor_radii,
        uncertainties_pos
    )

    uncertainty_safe = jnp.where(jnp.isnan(interpolated_uncertainty),
                                 jnp.ones_like(interpolated_uncertainty),
                                 1e-10 + interpolated_uncertainty)

    distance_error = jnp.abs((predicted_distances - true_distances))/uncertainty_safe
    distance_valid_mask = ~jnp.isnan(distance_error)
    distance_error_masked = jnp.sum(distance_error * distance_valid_mask, axis=-1)
    num_valid_distance_queries = jnp.sum(distance_valid_mask)

    eikonality_error = jnp.abs(
        1.0 - jnp.sqrt(jnp.sum(predicted_gradients * predicted_gradients, axis=-1))
    )
    eikonality_valid_mask = ~jnp.isnan(eikonality_error)
    eikonality_error_masked = jnp.sum(
        jnp.where(eikonality_valid_mask, eikonality_error, 0.0), axis=-1
    )
    num_valid_eikonality_queries = jnp.sum(eikonality_valid_mask)

    distance_loss = jax.lax.cond(
        num_valid_distance_queries > 0,
        lambda: jnp.sum(distance_error_masked) / num_valid_distance_queries,
        lambda: 0.0,
    )
    eikonality_loss = jax.lax.cond(
        num_valid_eikonality_queries > 0,
        lambda: jnp.sum(eikonality_error_masked) / num_valid_eikonality_queries,
        lambda: 0.0,
    )

    uncertainty_loss = 0.5*jnp.mean(uncertainty_safe**2 - 2*jnp.log(uncertainty_safe) - 1)

    loss = distance_loss + eikonality_loss + \
        uncertainty_coefficient*uncertainty_loss

    loss_dict = {
        'distance_loss': distance_loss,
        'eikonality_loss': eikonality_loss,
        'uncertainty_loss': jnp.mean(uncertainty_safe)
    }
    return loss, loss_dict


# ---------------------------------------------------------------------------
# Model / optimizer / train_step factory
# ---------------------------------------------------------------------------
def FundamentalFormPredictorTrainer(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    total_steps: int,
    lambda_coefficient: float,
    uncertainty_coefficient: float,
    rngs: nnx.Rngs
):
    model = FundamentalFormPredictorWithUncertainty(
        model_config,
        rngs=rngs,
    )

    learning_rate = training_config.learning_rate
    warmup_steps = training_config.warmup_steps

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-4*learning_rate,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=1e-4*learning_rate,
    )

    optimizer = nnx.Optimizer(
        model,
        optax.chain(optax.adam(learning_rate=schedule)),
        wrt=nnx.Param,
    )

    @nnx.jit
    def train_step(
        model: FundamentalFormPredictorWithUncertainty,
        optimizer: nnx.Optimizer,
        mask,
        positions,
        normals,
        neighbors,
        queries,
        true_distances
    ):
        """
        Single training step on a single batch (neighborhood) of one shape.

        Args:
            mask: (n_devices, n_points)
            positions, normals: (n_devices, n_points, 3)
            neighbors: (n_devices, n_points, k)
            queries: (n_devices, pointcloud_size, batch_size, 3)
            true_distances: (n_devices, batch_size)

        Returns:
            Loss averaged over batch.
        """

        queries = queries.reshape((queries.shape[0], -1, 3))
        true_distances = true_distances.reshape((true_distances.shape[0], -1))

        def loss_fn(model):
            positions_for_model = jax.vmap(lambda x, i: x[i])(positions, neighbors)
            normals_for_model = jax.vmap(lambda x, i: x[i])(normals, neighbors)
            model_out = model(positions_for_model, normals_for_model)
            uncertainty = model_out[..., 6]
            coeffs = model_out[..., :6]

            loss, loss_dict = SDF_loss_with_uncertainty(
                lambda_coefficient,
                uncertainty_coefficient,
                mask,
                positions,
                normals,
                coeffs,
                uncertainty,
                queries,
                true_distances
            )

            return loss, loss_dict

        (batch_loss, loss_dict), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

        def zero_out_nans(x):
            if x is None:
                return x
            return jnp.where(jnp.isnan(x), 0.0, x)

        # Apply to all gradients in the tree
        grads = jax.tree_util.tree_map(zero_out_nans, grads)

        # Update with this batch's gradients
        optimizer.update(model, grads)

        return batch_loss, loss_dict

    return model, optimizer, train_step


def load_all_training_data(training_data_filenames):
    all_training_data = []
    n_total_points = 0
    t0 = time.time()
    k_neighbors = None
    for training_data_filename in training_data_filenames:
        if not os.path.exists(training_data_filename):
            raise ValueError(f"The path '{training_data_filename}' does not exist")
        print(f"Loading training data from '{training_data_filename}'...")
        data = load_mesh_training_data(training_data_filename)
        (all_positions, all_normals, all_neighbors, all_masks,
         all_mesh_vertices, all_mesh_faces, all_face_masks) = data
        n_total_points += len(all_positions)
        k_neighbors = all_neighbors.shape[2]
        n_shapes = len(all_positions)
        pointcloud_size = all_positions.shape[1]
        print(f'  Loaded {n_shapes} shapes with max {pointcloud_size} points each')
        print(f'  k_neighbors: {k_neighbors}')
        all_training_data.append(data)
    t1 = time.time()
    print(f'Load data: {t1 - t0:.6f} s')

    return pointcloud_size, all_training_data


def load_validation_data():
    val_data_filename = 'datasets/ABC17_validation_3.npz'
    if os.path.exists(val_data_filename):
        try:
            val_data = load_mesh_training_data(val_data_filename)
            print(f'Loaded validation data: {len(val_data[0])} shapes')
            return val_data
        except Exception as e:
            print(f'Warning: Could not load validation data: {e}')
    else:
        print(f'Validation file not found: {val_data_filename}')


def train_fundamental_form_predictor(
    model_dir: str,
    seed: int,
    training_data_filenames: List[str],
    model_config: ModelConfig,
    training_config: TrainingConfig,
    lambda_coefficient: float,
    uncertainty_coefficient: float,
    load_model_path: str = None,
):
    LABEL_WIDTH = 18
    VALUE_WIDTH = 12

    assert training_config.queries_per_neighborhood % 3 == 0

    key = jax.random.PRNGKey(seed)
    rngs = nnx.Rngs(seed)

    mesh, model_sharding, data_sharding = build_device_mesh()
    n_devices = mesh.devices.size

    # load data
    pointcloud_size, all_training_data = load_all_training_data(training_data_filenames)
    (val_positions, val_normals, val_neighbors, val_masks,
                 val_mesh_vertices, val_mesh_faces, val_face_masks) = load_validation_data()

    # ---- Shape index bookkeeping ----
    all_shape_indices = []
    for dataset_idx, data in enumerate(all_training_data):
        n_shapes = len(data[0])
        for shape_idx in range(n_shapes):
            all_shape_indices.append((dataset_idx, shape_idx))
    total_shapes = len(all_shape_indices)
    print(f'Total shapes across all datasets: {total_shapes}')

    # initialize model and optimizer
    total_steps = 1 + (training_config.num_epochs*total_shapes) // (n_devices)
    model, optimizer, train_step = FundamentalFormPredictorTrainer(
        model_config,
        training_config,
        total_steps,
        lambda_coefficient,
        uncertainty_coefficient,
        rngs
    )

    if load_model_path is not None:
        model = model.restore_from_checkpoint(os.path.abspath(load_model_path))
        print(f'Warm-started from {load_model_path}')

    # ---- Shard model + optimizer state ----
    state = nnx.state((model, optimizer))
    state = jax.device_put(state, model_sharding)
    nnx.update((model, optimizer), state)

    # ---- Training loop ----
    model.train()

    t0 = time.time()
    all_train_losses = []
    all_distance_losses = []
    all_eikonality_losses = []
    validation_losses = []
    validation_epochs = []

    print('\nBeginning training...')
    print(f'  Batch size (per shape): {pointcloud_size*training_config.queries_per_neighborhood}')
    print(f'  Shapes per step (across GPUs): {n_devices}')

    np.random.seed(seed)

    for epoch in range(training_config.num_epochs):
        epoch_key = jax.random.fold_in(key, epoch)
        epoch_losses = []
        epoch_distance_losses = []
        epoch_eikonality_losses = []
        epoch_uncertainty_losses = []

        shuffled_indices = np.random.permutation(total_shapes)

        # Process shapes in groups of n_devices
        n_groups = total_shapes // n_devices
        for group_i in range(n_groups):
            group_indices = shuffled_indices[
                group_i * n_devices : (group_i + 1) * n_devices
            ]

            # ---- Prepare per-shape arrays for this group ----
            # We must generate queries per-shape, then stack along axis 0.
            # To keep memory bounded, we first build a list of per-shape dicts.
            per_shape = []
            for global_idx in group_indices:
                dataset_idx, shape_idx = all_shape_indices[global_idx]
                (all_positions, all_normals, all_neighbors, all_masks,
                 all_mesh_vertices, all_mesh_faces, all_face_masks) = \
                    all_training_data[dataset_idx]

                shape_key = jax.random.fold_in(epoch_key, int(global_idx))
                positions_np = all_positions[shape_idx]
                normals_np = all_normals[shape_idx]
                neighbors_np = all_neighbors[shape_idx]
                mask_np = all_masks[shape_idx]
                mesh_vertices_np = all_mesh_vertices[shape_idx]
                mesh_faces_np = all_mesh_faces[shape_idx]
                face_mask_np = all_face_masks[shape_idx]

                # Move to device for query generation
                positions_d = jax.device_put(positions_np, model_sharding)
                normals_d = jax.device_put(normals_np, model_sharding)
                neighbors_d = jax.device_put(neighbors_np, model_sharding)
                mask_d = jax.device_put(mask_np, model_sharding)
                mv_d = jax.device_put(mesh_vertices_np, model_sharding)
                mf_d = jax.device_put(mesh_faces_np, model_sharding)
                fm_d = jax.device_put(face_mask_np, model_sharding)

                shape_queries, shape_distances = generate_shape_features(
                    positions_d,
                    neighbors_d,
                    mv_d,
                    mf_d,
                    fm_d,
                    shape_key,
                    n_queries=training_config.queries_per_neighborhood,
                )

                per_shape.append(dict(
                    positions=np.asarray(positions_d),
                    normals=np.asarray(normals_d),
                    neighbors=np.asarray(neighbors_d),
                    mask=np.asarray(mask_d),
                    queries=np.asarray(shape_queries),
                    distances=np.asarray(shape_distances),
                    global_idx=int(global_idx),
                ))

            stacked = {}
            for key_name, np_key in [
                ('positions', 'positions'),
                ('normals',   'normals'),
                ('neighbors', 'neighbors'),
                ('mask',      'mask'),
                ('queries',   'queries'),
                ('distances', 'distances'),
            ]:
                arrs = []
                for p in per_shape:
                    arr = p[np_key]
                    arrs.append(arr)
                stacked[key_name] = np.stack(arrs, axis=0)

            batch_positions = jax.device_put(
                stacked['positions'], data_sharding
            )
            batch_normals = jax.device_put(
                stacked['normals'], data_sharding
            )
            mask_stack = jax.device_put(
                stacked['mask'], data_sharding
            )
            batch_neighbors = jax.device_put(
                stacked['neighbors'], data_sharding
            )
            batch_queries = jax.device_put(
                stacked['queries'],
                data_sharding,
            )
            batch_distances = jax.device_put(
                stacked['distances'],
                data_sharding,
            )

            loss, loss_dict = train_step(
                model, optimizer,
                mask_stack,
                batch_positions, batch_normals, batch_neighbors,
                batch_queries, batch_distances,
            )

            # All-reduce is implicit in JIT; loss is already global
            epoch_losses.append(float(jax.device_get(loss)))
            epoch_distance_losses.append(
                float(jax.device_get(loss_dict['distance_loss']))
            )
            epoch_eikonality_losses.append(
                float(jax.device_get(loss_dict['eikonality_loss']))
            )
            epoch_uncertainty_losses.append(
                float(jax.device_get(loss_dict['uncertainty_loss']))
            )

            if group_i % training_config.print_every == 0 and group_i > 0:
                epoch_train_loss = np.mean(epoch_losses)
                epoch_distance_loss = np.mean(epoch_distance_losses)
                epoch_eikonality_loss = np.mean(epoch_eikonality_losses)
                epoch_uncertainty_loss = np.mean(epoch_uncertainty_losses)
                print(
                    f'  Epoch {epoch}, group {group_i}/{n_groups}: '
                    f'loss = {epoch_train_loss:.6f}  '
                    f'distance = {epoch_distance_loss:.6f}  '
                    f'eikonality = {epoch_eikonality_loss:.6f}  '
                    f'uncertainty = {epoch_uncertainty_loss:.6f}'
                )
        epoch_train_loss = np.mean(epoch_losses)
        epoch_distance_loss = np.mean(epoch_distance_losses)
        epoch_eikonality_loss = np.mean(epoch_eikonality_losses)
        epoch_uncertainty_loss = np.mean(epoch_uncertainty_losses)

        all_train_losses.append(epoch_train_loss)
        all_distance_losses.append(epoch_distance_loss)
        all_eikonality_losses.append(epoch_eikonality_loss)

        if epoch % training_config.save_and_validate_every == 0:
            print(
                f'Epoch {epoch}:\n'
                f'    {"total loss:":<{LABEL_WIDTH}}{epoch_train_loss:<{VALUE_WIDTH}.6f}'
                f'{"distance:":<{LABEL_WIDTH}}{epoch_distance_loss:<{VALUE_WIDTH}.6f}'
                f'{"eikonality:":<{LABEL_WIDTH}}{epoch_eikonality_loss:<{VALUE_WIDTH}.6f}  '
                f'uncertainty = {epoch_uncertainty_loss:.6f}'
            )
            model.save_to_checkpoint(os.path.abspath(f'{model_dir}_{epoch}'))

            # ---- Validation (kept single-device for simplicity) ----
            if val_positions is not None:
                model.eval()
                n_val_shapes = len(val_positions)
                val_shape_losses = []
                val_shape_eikonality_losses = []
                val_shape_uncertainty_losses = []

                for val_shape_idx in range(n_val_shapes):
                    val_key = jax.random.fold_in(epoch_key, 10000 + val_shape_idx)
                    val_pos = val_positions[val_shape_idx]
                    val_norm = val_normals[val_shape_idx]
                    val_neigh = val_neighbors[val_shape_idx]
                    val_mask = val_masks[val_shape_idx]
                    val_verts = val_mesh_vertices[val_shape_idx]
                    val_faces = val_mesh_faces[val_shape_idx]
                    val_fmask = val_face_masks[val_shape_idx]

                    val_queries, val_distances = generate_shape_features(
                        val_pos,
                        val_neigh,
                        val_verts,
                        val_faces,
                        val_fmask,
                        val_key,
                        n_queries=training_config.queries_per_neighborhood,
                    )

                    val_output = model(val_pos[val_neigh][jnp.newaxis], val_norm[val_neigh][jnp.newaxis])
                    val_coeffs = val_output[..., :6]
                    val_uncertainty = val_output[..., 6]

                    batch_val_losses = []
                    batch_val_eikonality_losses = []
                    batch_val_uncertainty_losses = []

                    if jnp.sum(val_mask) > 0:
                        vl, val_loss_dict = SDF_loss_with_uncertainty(
                            lambda_coefficient,
                            uncertainty_coefficient,
                            val_mask[jnp.newaxis],
                            val_pos[jnp.newaxis],
                            val_norm[jnp.newaxis],
                            val_coeffs,
                            val_uncertainty,
                            val_queries.reshape((1, -1, 3)),
                            val_distances.reshape((1, -1,))
                        )
                        batch_val_losses.append(float(vl))
                        batch_val_eikonality_losses.append(float(val_loss_dict['eikonality_loss']))
                        batch_val_uncertainty_losses.append(float(val_loss_dict['uncertainty_loss']))

                    if len(batch_val_losses) > 0:
                        val_shape_losses.append(np.mean(batch_val_losses))
                        val_shape_eikonality_losses.append(
                            np.mean(batch_val_eikonality_losses)
                        )
                        val_shape_uncertainty_losses.append(
                            np.mean(batch_val_uncertainty_losses)
                        )

                val_loss = np.mean(val_shape_losses) if val_shape_losses else 0.0
                val_eik  = np.mean(val_shape_eikonality_losses) if val_shape_eikonality_losses else 0.0

                validation_losses.append(val_loss)
                validation_epochs.append(epoch)
                print(
                    f'    {"validation:":<{LABEL_WIDTH}}{val_loss:<{VALUE_WIDTH}.6f}'
                    f'{"eikonality:":<{LABEL_WIDTH}}{val_eik:<{VALUE_WIDTH}.6f}'
                    f'uncertainty = {epoch_uncertainty_loss:.6f}'
                )
                model.train()

    t1 = time.time()
    print(f'This phase training time: {t1 - t0:.2f} s')

    model.save_to_checkpoint(os.path.abspath(model_dir))


if __name__ == '__main__':
    dataset_dir = 'datasets/nice'

    parser = ArgumentParser()
    parser.add_argument("--config_file", type=str, default="macoroni_config.yaml")
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config_dict = yaml.safe_load(f)

    experiment_name = config_dict['experiment_name']
    experiment_dir = os.path.join('models', experiment_name)

    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config_dict, f)

    if os.path.exists(experiment_dir):
        print(f"WARNING: EXPERIMENT DIRECTORY {experiment_dir} ALREADY EXISTS!!! "
              "ASSUMING THIS IS INTENTIONAL AND CONTINUING!!!")
    else:
        os.makedirs(experiment_dir)

    seed = config_dict['seed']

    model_config = ModelConfig(
        **config_dict['model_config']
    )

    lambda_coefficient = config_dict['lambda_coefficient']
    uncertainty_coefficient = config_dict['uncertainty_coefficient']

    t0 = time.time()

    # Pretraining: Fit 10 tori
    pretraining_model_dir = f'{experiment_dir}/{experiment_name}_pretraining'
    if config_dict['run_pretraining']:
        pretraining_data_filenames = ["datasets/tori_meshes.npz"]
        pretraining_config = TrainingConfig(
            **config_dict['pretraining_config']
        )
        print(f"Pretraining config: {pretraining_config}")
        train_fundamental_form_predictor(
            pretraining_model_dir,
            seed,
            pretraining_data_filenames,
            model_config,
            pretraining_config,
            lambda_coefficient,
            uncertainty_coefficient,
        )

    # Phase 0: Nicely sampled blobs
    phase0_model_dir = f'{experiment_dir}/{experiment_name}_phase0'
    if config_dict['run_phase0']:
        phase0_data_filenames = sorted(glob.glob(os.path.join(dataset_dir, 'blobs_train_mode0*.npz')))
        phase0_data_filenames.append("datasets/tori_meshes.npz")
        phase0_config = TrainingConfig(
            **config_dict['phase0_config']
        )
        print(f"Phase 0 config: {phase0_config}")
        train_fundamental_form_predictor(
            phase0_model_dir,
            seed,
            phase0_data_filenames,
            model_config,
            phase0_config,
            lambda_coefficient,
            uncertainty_coefficient,
            load_model_path=pretraining_model_dir
        )

    # Phase 1: Nicely sampled blobs and ABC meshes
    phase1_model_dir = f'{experiment_dir}/{experiment_name}_phase1'
    if config_dict['run_phase1']:
        blob_filenames = glob.glob(os.path.join(dataset_dir, 'blobs_train_mode0*.npz'))
        ABC_filenames = glob.glob(os.path.join(dataset_dir, 'ABC17_train_mode0_*.npz'))
        lists = [blob_filenames, ABC_filenames]
        phase1_data_filenames = [val for tup in zip(*lists) for val in tup]  # interleave lists
        phase1_data_filenames.append("datasets/tori_meshes.npz")
        phase1_config = TrainingConfig(
            **config_dict['phase1_config']
        )
        print(f"Phase 1 config: {phase1_config}")
        train_fundamental_form_predictor(
            phase1_model_dir,
            seed,
            phase1_data_filenames,
            model_config,
            phase1_config,
            lambda_coefficient,
            uncertainty_coefficient,
            load_model_path=phase0_model_dir
        )

    # Phase 2: nicely sampled blobs and more ABC meshes
    phase2_model_dir = f'{experiment_dir}/{experiment_name}_phase2'
    if config_dict['run_phase2']:
        blob_filenames = glob.glob(os.path.join(dataset_dir, 'blobs_train_mode0*.npz'))
        ABC17_filenames = glob.glob(os.path.join(dataset_dir, 'ABC17_train_mode0_*.npz'))
        ABC2_filenames = glob.glob(os.path.join(dataset_dir, 'ABC2_train_mode0_*.npz'))
        phase2_data_filenames = blob_filenames + ABC17_filenames + ABC2_filenames + ["datasets/tori_meshes.npz"]
        phase2_config = TrainingConfig(
            **config_dict['phase2_config']
        )
        print(f"Phase 2 config: {phase2_config}")
        train_fundamental_form_predictor(
            phase2_model_dir,
            seed,
            phase2_data_filenames,
            model_config,
            phase2_config,
            lambda_coefficient,
            uncertainty_coefficient,
            load_model_path=phase1_model_dir
        )

    # Phase 3: nicely sampled blobs and even more ABC meshes
    phase3_model_dir = f'{experiment_dir}/{experiment_name}_phase3'
    if config_dict['run_phase3']:
        blob_filenames = glob.glob(os.path.join(dataset_dir, 'blobs_train_mode0*.npz'))
        ABC_filenames = glob.glob(os.path.join(dataset_dir, 'ABC*_train_mode0_*.npz'))
        phase3_data_filenames = blob_filenames + ABC_filenames + ["datasets/tori_meshes.npz"]
        phase3_config = TrainingConfig(
            **config_dict['phase3_config']
        )
        print(f"Phase 3 config: {phase3_config}")
        train_fundamental_form_predictor(
            phase3_model_dir,
            seed,
            phase3_data_filenames,
            model_config,
            phase3_config,
            lambda_coefficient,
            uncertainty_coefficient,
            load_model_path=phase2_model_dir
        )

    # Phase 4: everything from before, plus unevenly sampled data possibly with small gaps from a simulated scanning process
    phase4_model_dir = f'{experiment_dir}/{experiment_name}_phase4'
    if config_dict['run_phase4']:
        blob_filenames = glob.glob(os.path.join(dataset_dir, 'blobs_train_mode0*.npz'))
        ABC_filenames = glob.glob(os.path.join(dataset_dir, 'ABC*_train_mode0_*.npz'))
        blob_filenames_simulated = glob.glob(os.path.join('datasets/simulated', 'blobs_train_mode*.npz'))
        ABC_filenames_simulated = glob.glob(os.path.join('datasets/simulated', 'ABC*_train_mode*.npz'))
        phase4_data_filenames = blob_filenames + ABC_filenames + blob_filenames_simulated + ABC_filenames_simulated + ["datasets/tori_meshes.npz"]
        phase4_config = TrainingConfig(
            **config_dict['phase4_config']
        )
        print(f"Phase 4 config: {phase4_config}")
        train_fundamental_form_predictor(
            phase4_model_dir,
            seed,
            phase4_data_filenames,
            model_config,
            phase4_config,
            lambda_coefficient,
            uncertainty_coefficient,
            load_model_path=phase3_model_dir
        )

    t1 = time.time()
    print(f"Total training time: {t1 - t0} seconds")
import os
import glob
import time
import pickle

import numpy as np
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx
from jax.sharding import PartitionSpec, NamedSharding
from jax.experimental import mesh_utils

import matplotlib.pyplot as plt

from pointsastori.shape_3d import *
from pointsastori.network import *
from mesh_data import *
from mesh_sampling_jax import *
from train import load_mesh_training_data


# ---------------------------------------------------------------------------
# Device mesh (built lazily so importing this module doesn't force init)
# ---------------------------------------------------------------------------
def build_mesh():
    num_devices = jax.local_device_count()
    assert num_devices >= 1
    mesh = jax.sharding.Mesh(
        mesh_utils.create_device_mesh((num_devices,)),
        ('data',),
    )
    model_sharding = NamedSharding(mesh, PartitionSpec())       # replicated
    data_sharding  = NamedSharding(mesh, PartitionSpec('data')) # sharded on axis 0
    return mesh, model_sharding, data_sharding


# ---------------------------------------------------------------------------
# Losses (unchanged except they now expect axis-0 = "shape slot")
# ---------------------------------------------------------------------------
@jax.jit
def SDF_loss_with_uncertainty(
    mask, batch_mask, positions, normals,
    coefficients, uncertainties, queries, true_distances,
    neighborhood_radii
):
    uncertainties_pos = nnx.softplus(uncertainties)

    centers, axes, major_radii, minor_radii = torus_precompute_masked_sharded(
        mask, positions, normals, coefficients
    )
    predicted_distances, predicted_gradients, interpolated_uncertainty, weight, normalization = torus_distance_gradient_masked_with_clamp(
        mask, queries, positions, centers, axes, major_radii, minor_radii, neighborhood_radii, uncertainties_pos
    )

    uncertainty_safe = 1e-10 + interpolated_uncertainty

    distance_error = jnp.abs((predicted_distances - true_distances))/uncertainty_safe
    distance_valid_mask = ~jnp.isnan(distance_error) & batch_mask[:, :, None]
    distance_error_masked = jnp.sum(distance_error * distance_valid_mask, axis=-1)
    num_valid_distance_queries = jnp.sum(distance_valid_mask)

    eikonality_error = jnp.abs(
        1.0 - jnp.sqrt(jnp.sum(predicted_gradients * predicted_gradients, axis=-1))
    )
    eikonality_valid_mask = ~jnp.isnan(eikonality_error) & batch_mask[:, :, None]
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

    uncertainty_loss = 0.5*jnp.max(uncertainty_safe - jnp.log(uncertainty_safe))

    loss = distance_loss + eikonality_loss + uncertainty_loss

    loss_dict = {
        'distance_loss': distance_loss,
        'eikonality_loss': eikonality_loss,
        'uncertainty_loss': jnp.mean(uncertainty_safe),
        'normalization': jnp.min(jnp.sum(weight/normalization, axis=-1))
    }
    return loss, loss_dict


@jax.jit
def validation_loss(
    mask, batch_mask, positions, normals,
    coefficients, uncertainties, queries, true_distances,
    neighborhood_radii
):
    uncertainties_pos = nnx.softplus(uncertainties)

    centers, axes, major_radii, minor_radii = torus_precompute_masked_sharded(
        mask, positions, normals, coefficients
    )
    predicted_distances, predicted_gradients, interpolated_uncertainty, _, _ = torus_distance_gradient_masked_with_clamp(
        mask, queries, positions, centers, axes, major_radii, minor_radii, neighborhood_radii, uncertainties_pos
    )

    uncertainty_safe = 1e-10 + interpolated_uncertainty

    distance_error = jnp.abs((predicted_distances - true_distances)) / uncertainty_safe
    distance_valid_mask = ~jnp.isnan(distance_error) & batch_mask[..., jnp.newaxis]
    distance_error_masked = jnp.sum(distance_error * distance_valid_mask, axis=-1)
    num_valid_distance_queries = jnp.sum(distance_valid_mask)

    eikonality_error = jnp.abs(
        1.0 - jnp.sqrt(jnp.sum(predicted_gradients * predicted_gradients, axis=-1))
    )
    eikonality_valid_mask = ~jnp.isnan(eikonality_error) & batch_mask[..., jnp.newaxis]
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

    uncertainty_loss = 0.5*jnp.mean(uncertainty_safe - jnp.log(uncertainty_safe))

    loss = distance_loss + eikonality_loss + uncertainty_loss

    loss_dict = {
        'distance_loss': distance_loss,
        'eikonality_loss': eikonality_loss,
        'uncertainty_loss': uncertainty_loss
    }
    return loss, loss_dict


# ---------------------------------------------------------------------------
# Model / optimizer / train_step factory
# ---------------------------------------------------------------------------
def FundamentalFormPredictorTrainer(
    model_name: str,
    embed_dim=128,
    n_freq=4,
    n_layers=4,
    n_heads=4,
    mlp_dim=256,
    dropout_rate=0.0,
    rngs=None,
    learning_rate=1e-3,
    weight_decay=0.0,
    warmup_steps=1000,
    total_steps=100000,
    seed=44,
):
    if rngs is None:
        rngs = nnx.Rngs(seed)

    model = FundamentalFormPredictorWithUncertainty(
        embed_dim=embed_dim,
        n_freq=n_freq,
        n_layers=n_layers,
        n_heads=n_heads,
        mlp_dim=mlp_dim,
        dropout_rate=dropout_rate,
        rngs=rngs,
    )

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=learning_rate * 0.001,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=learning_rate * 0.01,
    )

    optimizer = nnx.Optimizer(
        model,
        optax.chain(optax.adamw(learning_rate=schedule, weight_decay=weight_decay)),
        wrt=nnx.Param,
    )

    @nnx.jit
    def train_step(
        model: FundamentalFormPredictor,
        optimizer: nnx.Optimizer,
        mask,
        batch_mask,
        positions,
        normals,
        neighbors,
        queries,
        true_distances,
        neighborhood_radii
    ):
        """
        Single training step on a single batch (neighborhood) of one shape.

        Args:
            mask: (max_points,)
            batch_mask: (batch_size, )
            positions, normals: (|P|, 3)
            neighbors: (|P|, k)
            queries: (batch_size * n_queries, 3)
            true_distances: (batch_size * n_queries)

        Returns:
            Loss averaged over batch.
        """

        def loss_fn(model):
            positions_for_model = jax.vmap(lambda x, i: x[i])(positions, neighbors)
            normals_for_model = jax.vmap(lambda x, i: x[i])(normals, neighbors)
            model_out = model(positions_for_model, normals_for_model)
            uncertainty = model_out[..., -1]
            coeffs = model_out[..., :-1]

            loss, loss_dict = SDF_loss_with_uncertainty(
                mask, batch_mask, positions, normals, coeffs, uncertainty, queries, true_distances, neighborhood_radii
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


# ---------------------------------------------------------------------------
# Main tuning driver
# ---------------------------------------------------------------------------
def TrainFundamentalFormPredictor(
    model_name: str,
    training_data_filenames,
    num_epochs,
    learning_rate,
    batch_size,
    warmup_steps,
    n_epochs_to_save,
    n_shapes_to_save,
    drop_last_layer,
    load_model_path=None,
    final_save_filepath=None,
):
    LABEL_WIDTH = 18
    VALUE_WIDTH = 12

    print(jax.default_backend())
    print(jax.devices())

    mesh, model_sharding, data_sharding = build_mesh()
    n_devices = mesh.devices.size

    seed = 42
    key = jax.random.PRNGKey(seed)
    rngs = nnx.Rngs(seed)

    # Architecture
    embed_dim = 128
    n_freq = 0
    n_layers = 6
    n_heads = 8
    mlp_dim = embed_dim * 2

    # Sampling
    n_queries_per_neighborhood = 60
    narrow_band_offset = 0.2
    bbox_extension = 1.0

    dropout_rate = 0.0
    weight_decay = 0.0

    # ---- Load all training data ----
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
        max_points = all_positions.shape[1]
        print(f'  Loaded {n_shapes} shapes with max {max_points} points each')
        print(f'  k_neighbors: {k_neighbors}')
        all_training_data.append(data)
    t1 = time.time()
    print(f'Load data: {t1 - t0:.6f} s')

    # ---- Shape index bookkeeping ----
    all_shape_indices = []
    for dataset_idx, data in enumerate(all_training_data):
        n_shapes = len(data[0])
        for shape_idx in range(n_shapes):
            all_shape_indices.append((dataset_idx, shape_idx))
    total_shapes = len(all_shape_indices)
    print(f'Total shapes across all datasets: {total_shapes}')

    total_steps = 1 + (max_points*n_queries_per_neighborhood*total_shapes) // batch_size

    model, optimizer, train_step = FundamentalFormPredictorTrainer(
        model_name,
        embed_dim=embed_dim,
        n_freq=n_freq,
        n_layers=n_layers,
        n_heads=n_heads,
        mlp_dim=mlp_dim,
        dropout_rate=dropout_rate,
        rngs=rngs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        seed=seed,
    )

    checkpointer = ocp.StandardCheckpointer()

    # ---- Warm start (load weights INTO existing model; do not rebind) ----
    if load_model_path is not None:
        graphdef, abstract_state = nnx.split(model)
        state_restored = checkpointer.restore(os.path.abspath(f'models/{model_name}/{load_model_path}'), abstract_state)
        model = nnx.merge(graphdef, state_restored)
        print(f'Warm-started from {load_model_path}')

    # ---- Shard model + optimizer state (replicated) ----
    state = nnx.state((model, optimizer))
    state = jax.device_put(state, model_sharding)
    nnx.update((model, optimizer), state)

    if total_shapes < n_devices:
        raise ValueError(
            f"Need at least {n_devices} shapes to run {n_devices}-way sharding, "
            f"got {total_shapes}"
        )

    # ---- Validation data ----
    val_data_filename = 'datasets/ABC17_validation_3.npz'
    val_positions = val_normals = val_neighbors = val_masks = None
    val_mesh_vertices = val_mesh_faces = val_face_masks = None
    if os.path.exists(val_data_filename):
        try:
            (val_positions, val_normals, val_neighbors, val_masks,
             val_mesh_vertices, val_mesh_faces, val_face_masks) = \
                load_mesh_training_data(val_data_filename)
            print(f'Loaded validation data: {len(val_positions)} shapes')
        except Exception as e:
            print(f'Warning: Could not load validation data: {e}')
    else:
        print(f'Validation file not found: {val_data_filename}')

    def save_model(curr_model, save_filepath: str):
        _, state = nnx.split(curr_model)
        path = os.path.join(os.getcwd(), f'models/{model_name}/{save_filepath}')
        checkpointer.save(path, state)
    """
        curr_model.eval()

        # Extract only data (parameters), do not save graphdef
        state = nnx.state(curr_model)  # or use nnx.state(curr_model, nnx.Param)
        # Convert to pure dictionary (containing only arrays)
        #pure_state = nnx.to_pure_dict(state)

        saved_data = {
            'state': state,  # Pure dict, no functions
            'embed_dim': embed_dim,
            'n_freq': n_freq,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'mlp_dim': mlp_dim,
            'dropout_rate': dropout_rate,
        }
        if epoch == num_epochs - 1:
            path = save_filepath + '.pkl'
        else:
            path = save_filepath + f'_{epoch}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(saved_data, f)
        print(f'Model saved to {save_filepath}')
    """

    # ---- Training loop ----
    model.train()

    t0 = time.time()
    all_train_losses = []
    all_distance_losses = []
    all_eikonality_losses = []
    validation_losses = []
    validation_epochs = []

    print('\nBeginning training...')
    print(f'  Batch size (per shape): {batch_size}')
    print(f'  Shapes per step (across GPUs): {n_devices}')

    np.random.seed(seed)

    for epoch in range(num_epochs):
        epoch_key = jax.random.fold_in(key, epoch)
        epoch_losses = []
        epoch_distance_losses = []
        epoch_eikonality_losses = []
        epoch_uncertainty_losses = []
        total_num_batches = 0

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
            n_batches_max = 0
            for global_idx in group_indices:
                dataset_idx, shape_idx = all_shape_indices[global_idx]
                (all_positions, all_normals, all_neighbors, all_masks,
                 all_mesh_vertices, all_mesh_faces, all_face_masks) = \
                    all_training_data[dataset_idx]

                max_points = all_positions.shape[1]
                n_batches = (max_points + batch_size - 1) // batch_size
                n_batches_max = max(n_batches_max, n_batches)

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
                normals_d   = jax.device_put(normals_np, model_sharding)
                neighbors_d = jax.device_put(neighbors_np, model_sharding)
                mask_d      = jax.device_put(mask_np, model_sharding)
                mv_d = jax.device_put(mesh_vertices_np, model_sharding)
                mf_d = jax.device_put(mesh_faces_np, model_sharding)
                fm_d = jax.device_put(face_mask_np, model_sharding)

                # I added this because I am changing the SDF to account for the maximum distance
                # between a point and the rest of its neighborhood
                neighborhood_differences = positions_d[:, jnp.newaxis] - positions_d[neighbors_d]
                neighborhood_distances = jnp.sqrt(jnp.sum(neighborhood_differences**2, axis=-1))
                neighborhood_radii = jnp.max(neighborhood_distances, axis=-1)

                shape_queries, shape_distances = generate_shape_features(
                    positions_d, neighbors_d, mv_d, mf_d, fm_d,
                    shape_key,
                    n_queries=n_queries_per_neighborhood,
                    narrow_band_offset=narrow_band_offset,
                    bbox_extension=bbox_extension,
                )

                per_shape.append(dict(
                    positions=np.asarray(positions_d),
                    normals=np.asarray(normals_d),
                    neighbors=np.asarray(neighbors_d),
                    mask=np.asarray(mask_d),
                    queries=np.asarray(shape_queries),
                    distances=np.asarray(shape_distances),
                    neighborhood_radii=np.asarray(neighborhood_radii),
                    n_batches=n_batches,
                    global_idx=int(global_idx),
                ))
            n_batches_common = min(p['n_batches'] for p in per_shape)

            stacked = {}
            for key_name, np_key in [
                ('positions', 'positions'),
                ('normals',   'normals'),
                ('neighbors', 'neighbors'),
                ('mask',      'mask'),
                ('queries',   'queries'),
                ('distances', 'distances'),
                ('neighborhood_radii', 'neighborhood_radii'),
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

            for batch_idx in range(n_batches_common):
                start_idx = batch_idx * batch_size

                # neighbors/queries/distances are batch-specific
                batch_neighbors = jax.device_put(
                    stacked['neighbors'][:, start_idx:start_idx + batch_size],
                    data_sharding,
                )
                batch_queries = jax.device_put(
                    stacked['queries'][:, start_idx:start_idx + batch_size],
                    data_sharding,
                )
                batch_distances = jax.device_put(
                    stacked['distances'][:, start_idx:start_idx + batch_size],
                    data_sharding,
                )
                batch_mask = jax.device_put(
                    stacked['mask'][:, start_idx:start_idx + batch_size],
                    data_sharding,
                )
                batch_radii = jax.device_put(
                    stacked['neighborhood_radii'][:, start_idx:start_idx + batch_size],
                    data_sharding,
                )

                loss, loss_dict = train_step(
                    model, optimizer,
                    mask_stack, batch_mask,
                    batch_positions, batch_normals, batch_neighbors,
                    batch_queries, batch_distances,
                    batch_radii
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
                #print(f"normalization: {loss_dict['normalization']}")
                total_num_batches += 1

            if group_i % n_shapes_to_save == 0 and group_i > 0:
                epoch_train_loss = np.mean(epoch_losses) if epoch_losses else 0.0
                epoch_distance_loss = np.mean(epoch_distance_losses) if epoch_distance_losses else 0.0
                epoch_eikonality_loss = np.mean(epoch_eikonality_losses) if epoch_eikonality_losses else 0.0
                epoch_uncertainty_loss = np.mean(epoch_uncertainty_losses) if epoch_uncertainty_losses else 0.0
                print(
                    f'  Epoch {epoch}, group {group_i}/{n_groups}: '
                    f'loss = {epoch_train_loss:.6f}  '
                    f'distance = {epoch_distance_loss:.6f}  '
                    f'eikonality = {epoch_eikonality_loss:.6f}  '
                    f'uncertainty = {epoch_uncertainty_loss:.6f}'
                )
        epoch_train_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        epoch_distance_loss = np.mean(epoch_distance_losses) if epoch_distance_losses else 0.0
        epoch_eikonality_loss = np.mean(epoch_eikonality_losses) if epoch_eikonality_losses else 0.0
        epoch_uncertainty_loss = np.mean(epoch_uncertainty_losses) if epoch_uncertainty_losses else 0.0

        all_train_losses.append(epoch_train_loss)
        all_distance_losses.append(epoch_distance_loss)
        all_eikonality_losses.append(epoch_eikonality_loss)

        if epoch % n_epochs_to_save == 0:
            print(
                f'Epoch {epoch}:\n'
                f'    {"total loss:":<{LABEL_WIDTH}}{epoch_train_loss:<{VALUE_WIDTH}.6f}'
                f'{"distance:":<{LABEL_WIDTH}}{epoch_distance_loss:<{VALUE_WIDTH}.6f}'
                f'{"eikonality:":<{LABEL_WIDTH}}{epoch_eikonality_loss:<{VALUE_WIDTH}.6f}  '
                f'uncertainty = {epoch_uncertainty_loss:.6f}'
            )
            save_model(model, f'{final_save_filepath}_{epoch}')

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

                    # I added this because I am changing the SDF to account for the maximum distance
                    # between a point and the rest of its neighborhood
                    val_neighborhood_differences = val_pos[:, jnp.newaxis] - val_pos[val_neigh]
                    val_neighborhood_distances = jnp.sqrt(jnp.sum(val_neighborhood_differences**2, axis=-1))
                    val_neighborhood_radii = jnp.max(val_neighborhood_distances, axis=-1)

                    val_queries, val_distances = generate_shape_features(
                        val_pos, val_neigh, val_verts, val_faces, val_fmask,
                        val_key, n_queries=n_queries_per_neighborhood,
                    )
                    val_output = model(val_pos[val_neigh][jnp.newaxis], val_norm[val_neigh][jnp.newaxis])
                    val_coeffs = val_output[..., :-1]
                    val_uncertainty = val_output[..., -1]

                    n_val_batches = (val_pos.shape[0] + batch_size - 1) // batch_size
                    batch_val_losses = []
                    batch_val_eikonality_losses = []
                    batch_val_uncertainty_losses = []

                    for batch_idx in range(n_val_batches):
                        start_idx = batch_idx * batch_size
                        bm = jax.lax.dynamic_slice_in_dim(val_mask, start_idx, batch_size)
                        bq = jax.lax.dynamic_slice_in_dim(val_queries, start_idx, batch_size)
                        bd = jax.lax.dynamic_slice_in_dim(val_distances, start_idx, batch_size)
                        br = jax.lax.dynamic_slice_in_dim(val_neighborhood_radii, start_idx, batch_size)
                        if jnp.sum(bm) > 0:
                            vl, val_loss_dict = validation_loss(
                                val_mask[jnp.newaxis], bm[jnp.newaxis], val_pos[jnp.newaxis], val_norm[jnp.newaxis],
                                val_coeffs, val_uncertainty, bq[jnp.newaxis], bd[jnp.newaxis], br[jnp.newaxis]
                            )
                            batch_val_losses.append(float(vl))
                            batch_val_eikonality_losses.append(float(val_loss_dict['eikonality_loss']))
                            batch_val_uncertainty_losses.append(float(val_loss_dict['uncertainty_loss']))

                    if batch_val_losses:
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
    print(f'Total training time: {t1 - t0:.2f} s')

    train_losses = np.array(all_train_losses)
    distance_losses = np.array(all_distance_losses)
    eikonality_losses = np.array(all_eikonality_losses)

    save_model(model, f'{final_save_filepath}')

    # ---- Plot ----
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].plot(train_losses, label='Total Training Loss', color='blue')
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Training Loss'); axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(distance_losses, label='Distance Error', color='green')
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Distance Error'); axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(eikonality_losses, label='Eikonality Error', color='orange')
    axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Eikonality Error'); axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    if len(validation_losses) > 0:
        axes[1, 1].plot(validation_epochs, validation_losses,
                        label='Validation Loss', color='red', marker='o')
        axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Validation Loss'); axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('PlotTrainingHistory.png', dpi=150)
    print('Training history plot saved to PlotTrainingHistory.png')


def train():
    dataset_dir = 'datasets/nice'
    model_name = "UncertaintyTest1"

    # Phase 0: nicely sampled blobs
    training_data_filenames = sorted(glob.glob(os.path.join(dataset_dir, 'blobs_train_mode0*.npz')))
    TrainFundamentalFormPredictor(
        model_name,
        training_data_filenames,
        num_epochs=24,
        learning_rate=1e-3,
        warmup_steps=1e4,
        batch_size=2048,
        n_epochs_to_save=1,
        n_shapes_to_save=10,
        drop_last_layer=False,
        #load_model_path='models/FundamentalFormPredictor_pretrained.pkl',
        final_save_filepath=f'{model_name}_phase0',
    )

    # Phase 1: nicely sampled blobs and ABC meshes
    blob_filenames = glob.glob(os.path.join(dataset_dir, 'blobs_train_mode0*.npz'))
    ABC_filenames = glob.glob(os.path.join(dataset_dir, 'ABC17_train_mode0_*.npz'))
    lists = [blob_filenames, ABC_filenames]
    training_data_filenames = [val for tup in zip(*lists) for val in tup]  # interleave lists
    TrainFundamentalFormPredictor(
        model_name,
        training_data_filenames,
        num_epochs=8,
        learning_rate=1e-4,
        warmup_steps=1e4,
        batch_size=2048,
        n_epochs_to_save=1,
        n_shapes_to_save=100,
        drop_last_layer=False,
        load_model_path=f'{model_name}_phase0',
        final_save_filepath=f'{model_name}_phase1'
    )

    # Phase 2: nicely sampled blobs and more ABC meshes
    blob_filenames = glob.glob(os.path.join(dataset_dir, 'blobs_train_mode0*.npz'))
    ABC17_filenames = glob.glob(os.path.join(dataset_dir, 'ABC17_train_mode0_*.npz'))
    ABC2_filenames = glob.glob(os.path.join(dataset_dir, 'ABC2_train_mode0_*.npz'))
    training_data_filenames = blob_filenames + ABC17_filenames + ABC2_filenames
    TrainFundamentalFormPredictor(
        model_name,
        training_data_filenames,
        num_epochs=8,
        learning_rate=1e-5,
        warmup_steps=1e4,
        batch_size=2048,
        n_epochs_to_save=1,
        n_shapes_to_save=100,
        drop_last_layer=False,
        load_model_path=f'{model_name}_phase1',
        final_save_filepath=f'{model_name}_phase2',
    )

    # Phase 3: nicely sampled blobs and even more ABC meshes
    blob_filenames = glob.glob(os.path.join(dataset_dir, 'blobs_train_mode0*.npz'))
    ABC_filenames = glob.glob(os.path.join(dataset_dir, 'ABC*_train_mode0_*.npz'))
    training_data_filenames = blob_filenames + ABC_filenames
    TrainFundamentalFormPredictor(
        model_name,
        training_data_filenames,
        num_epochs=5,
        learning_rate=1e-6,
        warmup_steps=1e3,
        batch_size=2048,
        n_epochs_to_save=1,
        n_shapes_to_save=100,
        drop_last_layer=False,
        load_model_path=f'{model_name}_phase2',
        final_save_filepath=f'{model_name}_phase3',
    )

    # Phase 4: everything from before, plus unevenly sampled data possibly with small gaps from a simulated scanning process
    blob_filenames = glob.glob(os.path.join(dataset_dir, 'blobs_train_mode0*.npz'))
    ABC_filenames = glob.glob(os.path.join(dataset_dir, 'ABC*_train_mode0_*.npz'))
    blob_filenames_simulated = glob.glob(os.path.join('datasets/simulated', 'blobs_train_mode*.npz'))
    ABC_filenames_simulated = glob.glob(os.path.join('datasets/simulated', 'ABC*_train_mode*.npz'))
    training_data_filenames = blob_filenames + ABC_filenames + blob_filenames_simulated + ABC_filenames_simulated
    TrainFundamentalFormPredictor(
        model_name,
        training_data_filenames,
        num_epochs=2,
        learning_rate=1e-7,
        warmup_steps=1e3,
        batch_size=2048,
        n_epochs_to_save=1,
        n_shapes_to_save=100,
        drop_last_layer=False,
        load_model_path=f'{model_name}_phase3',
        final_save_filepath=f'{model_name}_phase4',
    )


if __name__ == '__main__':
    train()
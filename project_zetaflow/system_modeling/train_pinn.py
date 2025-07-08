# Copyright [2023] [Project ZetaFlow Contributors]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import mpmath # For generating accurate zeta values for training data
from typing import Tuple, Any, Callable, Optional

# JAX configuration for complex numbers
# JAX handles complex numbers natively in many operations.
# Ensure all custom layers or functions are compatible.

# Configuration
NUM_EPOCHS = 200
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
NUM_DATA_POINTS = 5000 # Number of s values for data loss
NUM_COLLOCATION_POINTS = 5000 # Number of s values for physics loss (functional equation)
PHYSICS_LOSS_WEIGHT = 0.1 # Weight for the functional equation residual in total loss

# Set mpmath precision
mpmath.mp.dps = 30 # Desired decimal places precision for zeta calculations

# --- PINN Model Definition (using Flax) ---

class ComplexDense(nn.Module):
	features: int
	kernel_init: Callable = nn.initializers.lecun_normal() # Good for SeLU/GeLU
	bias_init: Callable = nn.initializers.zeros

	@nn.compact
	def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
		"""Applies a dense layer to complex inputs."""
		# inputs shape: (batch, in_features) where elements are complex
		# We can implement complex dense layer by:
		# 1. Splitting real and imaginary parts and using two real dense layers.
		# 2. Or, if JAX/Flax Dense handles complex dtype directly with complex kernels.
		# JAX's Dense generally supports complex dtypes for inputs and kernels.

		# Ensure kernel and bias are complex
		kernel = self.param('kernel', self.kernel_init, (inputs.shape[-1], self.features), jnp.complex64)
		bias = self.param('bias', self.bias_init, (self.features,), jnp.complex64)

		return jnp.dot(inputs, kernel) + bias

class ComplexMLP(nn.Module):
	"""A simple MLP that handles complex numbers."""
	hidden_dims: Tuple[int, ...]
	output_dim: int # Output dim for complex numbers (e.g., 1 for zeta value)
	activation: Callable = nn.gelu # GeLU or SeLU can work well

	@nn.compact
	def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
		# x is complex, shape (batch, input_features)
		# Input to MLP is typically (real_part, imag_part) concatenated or handled as complex.
		# Here, x is a complex number s = sigma + it. We map it to (sigma, t).
		inp = jnp.stack([x.real, x.imag], axis=-1) # Shape (batch_size, 2)

		# Layers for processing real and imaginary parts separately or combined
		for dim in self.hidden_dims:
			inp = nn.Dense(features=dim)(inp)
			inp = self.activation(inp) # Apply activation to real-valued features

		# Output layer: map hidden state to real and imaginary parts of zeta(s)
		# We want one complex output, so two real outputs (real and imaginary parts)
		output_real = nn.Dense(features=self.output_dim, name="output_real")(inp)
		output_imag = nn.Dense(features=self.output_dim, name="output_imag")(inp)

		# Combine to complex output: zeta_real + 1j * zeta_imag
		# If output_dim is 1 (single complex value), shapes are (batch_size, 1)
		return output_real + 1j * output_imag


class ZetaPINN(nn.Module):
	"""
	Physics-Informed Neural Network for approximating Riemann Zeta function zeta(s).
	Input s is a complex number. Output is an approximation of zeta(s).
	"""
	hidden_dims: Tuple[int, ...] = (128, 128, 128) # Example architecture

	@nn.compact
	def __call__(self, s_complex: jnp.ndarray) -> jnp.ndarray:
		# s_complex is a JAX array of complex numbers, shape (batch,) or (batch, 1)
		# Ensure s_complex is reshaped if it's (batch,) for MLP input
		if s_complex.ndim == 1:
			s_complex_reshaped = s_complex[:, None] # (batch, 1)
		else:
			s_complex_reshaped = s_complex

		# The ComplexMLP expects input features for real and imag parts.
		# Here, s_complex_reshaped itself (if (batch,1)) is treated as a single complex feature.
		# The ComplexMLP internally splits it into real and imag parts for its first layer.
		zeta_approx_complex = ComplexMLP(hidden_dims=self.hidden_dims, output_dim=1)(s_complex_reshaped)

		return zeta_approx_complex.squeeze() # Return shape (batch,)


# --- Data Generation ---
def generate_training_data(num_data_points: int, num_col_points: int, key: jax.random.PRNGKey) -> Dict[str, jnp.ndarray]:
	"""
	Generates training data:
	- s_data: Points for data loss (where zeta(s) is known).
	- zeta_s_data: Corresponding known zeta(s) values.
	- s_collocation: Points for physics loss (functional equation).
	"""
	key_data, key_col = jax.random.split(key)

	# Generate s_data points in the critical strip (0 < Re(s) < 1) and some outside
	# For PINNs, it's good to sample boundary conditions and interior points.
	# Example: Sample sigma from U(0, 1) and t from U(0, 100)
	sigma_data = jax.random.uniform(key_data, (num_data_points,), minval=-2.0, maxval=3.0) # Wider range for stability
	t_data = jax.random.uniform(key_data, (num_data_points,), minval=0.1, maxval=100.0) # Avoid t=0 for some ops
	s_data = sigma_data + 1j * t_data

	# Generate s_collocation points for the functional equation
	# These can be sampled similarly or from a specific region of interest
	sigma_col = jax.random.uniform(key_col, (num_col_points,), minval=-5.0, maxval=6.0) # Wider range to test FE
	t_col = jax.random.uniform(key_col, (num_col_points,), minval=0.1, maxval=100.0)
	s_collocation = sigma_col + 1j * t_col

	# Compute zeta(s_data) using mpmath for ground truth
	print(f"Generating {num_data_points} ground truth zeta values using mpmath...")
	zeta_s_data_list = []
	for s_val in s_data:
		try:
			# mpmath handles complex numbers directly
			zeta_mp = mpmath.zeta(complex(s_val))
			zeta_s_data_list.append(jnp.complex64(zeta_mp))
		except Exception as e:
			print(f"Warning: mpmath.zeta computation failed for s={s_val}: {e}. Using NaN.")
			zeta_s_data_list.append(jnp.complex64(np.nan + 1j*np.nan))

	zeta_s_data = jnp.array(zeta_s_data_list)

	# Filter out NaNs that might have occurred if mpmath failed for some points
	# (e.g. s=1, or numerically difficult regions if precision is low)
	valid_indices = ~jnp.isnan(zeta_s_data.real) & ~jnp.isnan(zeta_s_data.imag)
	s_data = s_data[valid_indices]
	zeta_s_data = zeta_s_data[valid_indices]
	print(f"Generated {s_data.shape[0]} valid ground truth zeta values.")


	return {
		"s_data": s_data,
		"zeta_s_data": zeta_s_data,
		"s_collocation": s_collocation,
	}

# --- Loss Functions ---

def chi_function(s: jnp.ndarray) -> jnp.ndarray:
	"""
	Calculates the chi factor from the Riemann functional equation:
	zeta(s) = chi(s) * zeta(1-s)
	chi(s) = 2^s * pi^(s-1) * sin(pi*s/2) * Gamma(1-s)
	This requires a JAX-compatible Gamma function for complex inputs.
	JAX has `jax.scipy.special.gamma`.
	"""
	# Note: jax.scipy.special.gamma might not be as robust as mpmath.gamma for all complex inputs.
	# Consider pre-calculating or careful handling if issues arise.
	term1 = (2 * jnp.pi)**s
	term2 = jnp.sin(jnp.pi * s / 2.0) # sin should handle complex args
	# Gamma(1-s) can be problematic near poles.
	# Using loggamma might be more stable: Gamma(z) = exp(loggamma(z))
	# term3 = jax.scipy.special.gamma(1 - s) # Gamma(1-s)
	# More stable log-gamma formulation for chi(s) is often preferred.
	# chi(s) = pi^(s - 1/2) * Gamma((1-s)/2) / Gamma(s/2)
	# For now, using the direct form with jax.scipy.special.gamma:

	# Using the alternative form: chi(s) = pi^(s-1/2) * gamma((1-s)/2) / gamma(s/2)
	# This is often more numerically stable.
	log_gamma_half_s = jax.lax.lgamma(s / 2.0)
	log_gamma_half_one_minus_s = jax.lax.lgamma((1.0 - s) / 2.0)

	# chi_val = jnp.pi**(s - 0.5) * jnp.exp(log_gamma_half_one_minus_s - log_gamma_half_s)
	# The above is for xi(s) = xi(1-s) where xi(s) = s(s-1)/2 * pi^(-s/2) * Gamma(s/2) * zeta(s)
	# Let's use the form zeta(s) = 2^s * pi^(s-1) * sin(pi*s/2) * Gamma(1-s) * zeta(1-s)
	# So chi(s) = 2^s * pi^(s-1) * sin(pi*s/2) * Gamma(1-s)

	# Ensure gamma function is available and handles complex numbers
	# jax.scipy.special.gamma should work.
	gamma_one_minus_s = jax.scipy.special.gamma(1 - s)

	chi_val = (2**s) * (jnp.pi**(s - 1)) * jnp.sin(jnp.pi * s / 2.0) * gamma_one_minus_s
	return chi_val


def functional_equation_residual(zeta_s_pred: jnp.ndarray, zeta_one_minus_s_pred: jnp.ndarray, s_col: jnp.ndarray) -> jnp.ndarray:
	"""
	Calculates the residual of the Riemann functional equation:
	| zeta(s) - chi(s) * zeta(1-s) |^2
	Args:
		zeta_s_pred: Predicted zeta(s) by the PINN.
		zeta_one_minus_s_pred: Predicted zeta(1-s) by the PINN.
		s_col: Collocation points s.
	"""
	chi_s = chi_function(s_col)
	residual = zeta_s_pred - chi_s * zeta_one_minus_s_pred
	return jnp.abs(residual)**2 # Use squared magnitude

def total_loss_fn(
	params: Any, # Model parameters
	model_apply_fn: Callable,
	s_data: jnp.ndarray,
	zeta_s_data: jnp.ndarray, # Ground truth zeta(s_data)
	s_collocation: jnp.ndarray,
	physics_loss_weight: float
	) -> jnp.ndarray:
	"""Calculates the total loss (data loss + physics loss)."""

	# Data loss: MSE between predicted zeta(s_data) and actual zeta(s_data)
	zeta_s_data_pred = model_apply_fn({'params': params}, s_data)
	# Ensure predictions and targets have compatible shapes, e.g. (batch,)
	data_loss = jnp.mean(jnp.abs(zeta_s_data_pred.squeeze() - zeta_s_data.squeeze())**2)

	# Physics loss: Functional equation residual
	# Need zeta(s_col) and zeta(1-s_col) from the model
	zeta_s_col_pred = model_apply_fn({'params': params}, s_collocation)
	zeta_one_minus_s_col_pred = model_apply_fn({'params': params}, 1.0 - s_collocation)

	fe_residual = functional_equation_residual(zeta_s_col_pred, zeta_one_minus_s_col_pred, s_collocation)
	physics_loss = jnp.mean(fe_residual)

	total_loss = data_loss + physics_loss_weight * physics_loss
	return total_loss


# --- Training Loop ---
@jax.jit
def train_step_pinn(
	params: Any,
	opt_state: Any,
	s_data_batch: jnp.ndarray,
	zeta_s_data_batch: jnp.ndarray,
	s_col_batch: jnp.ndarray,
	model_apply_fn: Callable,
	optimizer_update_fn: Callable,
	loss_fn_for_grad: Callable # This will be `total_loss_fn` partially applied
	):
	"""Performs a single training step for the PINN."""

	# Compute loss and gradients
	loss_val, grads = jax.value_and_grad(loss_fn_for_grad)(params)

	# Update optimizer state and parameters
	updates, new_opt_state = optimizer_update_fn(grads, opt_state, params)
	new_params = optax.apply_updates(params, updates)

	return new_params, new_opt_state, loss_val


def train_pinn_model(
	pinn_model: ZetaPINN,
	training_data: Dict[str, jnp.ndarray],
	num_epochs: int,
	batch_size: int,
	learning_rate: float,
	physics_loss_weight: float,
	key: jax.random.PRNGKey
	):
	"""Main training loop for the Zeta PINN."""

	s_data = training_data["s_data"]
	zeta_s_data = training_data["zeta_s_data"]
	s_collocation = training_data["s_collocation"]

	# Initialize model parameters
	key_init, key_train_loop = jax.random.split(key)
	dummy_s_init = jnp.ones((1,), dtype=jnp.complex64) # Dummy input for initialization
	params = pinn_model.init(key_init, dummy_s_init)['params']

	# Initialize optimizer
	tx = optax.adam(learning_rate=learning_rate)
	opt_state = tx.init(params)

	# Create a partial function for the loss to pass to jax.grad
	# This captures model_apply_fn and other static args for total_loss_fn
	loss_fn_grad = jax.tree_util.Partial(
		total_loss_fn,
		model_apply_fn=pinn_model.apply,
		# s_data, zeta_s_data, s_collocation will be batched
		physics_loss_weight=physics_loss_weight
	)


	num_data_batches = s_data.shape[0] // batch_size
	num_col_batches = s_collocation.shape[0] // batch_size
	# Ensure num_batches is consistent or handle data padding/dropping last batch
	num_batches = min(num_data_batches, num_col_batches)


	print(f"Starting PINN training for {num_epochs} epochs...")
	for epoch in range(num_epochs):
		key_data_shuffle, key_col_shuffle, key_train_loop = jax.random.split(key_train_loop, 3)

		# Shuffle data
		perm_data = jax.random.permutation(key_data_shuffle, s_data.shape[0])
		shuffled_s_data = s_data[perm_data]
		shuffled_zeta_s_data = zeta_s_data[perm_data]

		perm_col = jax.random.permutation(key_col_shuffle, s_collocation.shape[0])
		shuffled_s_col = s_collocation[perm_col]

		epoch_loss = 0.0
		for batch_idx in range(num_batches):
			start_idx = batch_idx * batch_size
			end_idx = start_idx + batch_size

			s_data_batch = shuffled_s_data[start_idx:end_idx]
			zeta_s_data_batch = shuffled_zeta_s_data[start_idx:end_idx]
			s_col_batch = shuffled_s_col[start_idx:end_idx]

			# Update the arguments for the loss function for this batch
			batch_loss_fn_grad = jax.tree_util.Partial(
				loss_fn_grad, # This already has model.apply and physics_loss_weight
				s_data=s_data_batch,
				zeta_s_data=zeta_s_data_batch,
				s_collocation=s_col_batch
			)

			params, opt_state, loss_val = train_step_pinn(
				params, opt_state,
				s_data_batch, zeta_s_data_batch, s_col_batch, # these are now for info, not directly used by train_step_pinn
				pinn_model.apply, # model_apply_fn
				tx.update, # optimizer_update_fn
				batch_loss_fn_grad # The loss function that takes only params
			)
			epoch_loss += loss_val

		avg_epoch_loss = epoch_loss / num_batches
		print(f"Epoch {epoch+1}/{num_epochs}, Average Total Loss: {avg_epoch_loss:.6e}")

	print("PINN Training finished.")
	return params


# --- Main Execution ---
def main():
	print("Phase 3: System Modeling - Training Differentiable Zeta Surrogate (PINN)")
	key = jax.random.PRNGKey(42)

	# 1. Generate Training Data
	key_data_gen, key_training = jax.random.split(key)
	training_data = generate_training_data(NUM_DATA_POINTS, NUM_COLLOCATION_POINTS, key_data_gen)

	if training_data["s_data"].shape[0] == 0:
		print("No valid training data generated. Aborting.")
		return

	# 2. Initialize PINN Model
	pinn_model = ZetaPINN() # Using default hidden_dims

	# 3. Train the PINN Model
	trained_params = train_pinn_model(
		pinn_model,
		training_data,
		num_epochs=NUM_EPOCHS,
		batch_size=BATCH_SIZE,
		learning_rate=LEARNING_RATE,
		physics_loss_weight=PHYSICS_LOSS_WEIGHT,
		key=key_training
	)

	if trained_params:
		print("Zeta PINN model training complete. Trained parameters obtained.")
		# Save trained_params (e.g., using Flax serialization)
		# from flax.training import checkpoints
		# checkpoints.save_checkpoint(ckpt_dir='./pinn_checkpoints', target={'params': trained_params}, step=NUM_EPOCHS, overwrite=True)

		# Example: Test the trained model on a few points
		print("\nTesting trained PINN model:")
		test_points_s = jnp.array([0.5 + 14.1347j, 0.2 + 50j, 1.5 + 20j], dtype=jnp.complex64)

		# Apply function for inference
		@jax.jit
		def predict_zeta(params_inf, s_values):
			return pinn_model.apply({'params': params_inf}, s_values)

		predicted_zetas = predict_zeta(trained_params, test_points_s)

		for i, s_val_test in enumerate(test_points_s):
			actual_zeta_mp = mpmath.zeta(complex(s_val_test))
			print(f"s = {s_val_test:.4f}, PINN zeta(s) = {predicted_zetas[i]:.4f}, mpmath zeta(s) = {complex(actual_zeta_mp):.4f}")
	else:
		print("Zeta PINN model training failed.")


if __name__ == "__main__":
	main()

# Requirements:
# pip install jax jaxlib flax optax mpmath numpy scipy
# (scipy is for jax.scipy.special.gamma)
#
# Important Considerations:
# 1. Complex Number Handling: Ensure all layers and activations in `ComplexMLP`
#	are appropriate for complex numbers or operate on their real/imaginary parts correctly.
#	JAX's automatic differentiation handles complex numbers well (holomorphic functions).
# 2. Gamma Function: `jax.scipy.special.gamma` is used for `chi_function`. Its numerical
#	stability and accuracy for wide range of complex inputs should be monitored.
#	Using `jax.lax.lgamma` (log-gamma) is often more stable.
# 3. Training Data Distribution: The choice of `s_data` (for data loss) and `s_collocation`
#	(for physics loss) significantly impacts PINN performance. They should cover regions
#	of interest, including near the critical line, poles, zeros, and boundaries.
# 4. Loss Weights: `physics_loss_weight` is a hyperparameter. It balances fitting known data
#	versus satisfying the functional equation. This often requires tuning.
# 5. Model Architecture: The `hidden_dims` in `ZetaPINN` and `ComplexMLP` are examples.
#	Deeper/wider networks might be needed for higher accuracy.
# 6. Evaluation: Rigorous evaluation against known zeta values (not used in training data loss)
#	and checking the functional equation residual on a separate test set is crucial.
# 7. Numerical Precision: `mpmath.dps` controls precision for generating training data. JAX typically
#	uses float32 or float64. Ensure consistency or handle conversions carefully.
#	`jnp.complex64` is used in this script. For higher precision, JAX might need configuration
#	for float64 if not default (`jax.config.update("jax_enable_x64", True)`).
```

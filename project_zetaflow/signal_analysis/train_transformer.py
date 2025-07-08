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
import optax # For optimizers and learning rate schedules
from typing import List, Tuple, Dict, Any, Optional, Callable
import requests # To query Phase 1 API
from decimal import Decimal, getcontext

# Set precision for Decimal (if converting string zeros)
getcontext().prec = 50 # Set a high precision

# Configuration
PHASE1_API_URL = "http://localhost:8000" # Adjust if your Phase 1 API is elsewhere
DEFAULT_SEQUENCE_LENGTH = 128 # Max sequence length for Transformer input
DEFAULT_BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4

# --- Data Loading and Preprocessing ---
class ZetaZeroDataset:
	"""
	Dataset class to fetch Riemann zeta zeros from Phase 1 API,
	calculate normalized spacings, and prepare them for the Transformer model.
	"""
	def __init__(self, api_url: str, sequence_length: int):
		self.api_url = api_url
		self.sequence_length = sequence_length
		self.raw_zeros_t: List[Decimal] = [] # Store high-precision zeros (imaginary parts)

	def fetch_zeros(self, start_height_t: str = "0", end_height_t: str = "1000", limit: int = 10000) -> None:
		"""
		Fetches a batch of zeros from the Phase 1 API.
		Args:
			start_height_t: Minimum height T (as string for precision).
			end_height_t: Maximum height T (as string for precision).
			limit: Maximum number of zeros to fetch.
		"""
		endpoint = f"{self.api_url}/zeros/range"
		params = {
			"start_height_t": start_height_t,
			"end_height_t": end_height_t,
			"limit": limit,
			"offset": 0 # Adjust for pagination if needed
		}
		try:
			response = requests.get(endpoint, params=params)
			response.raise_for_status() # Raise an exception for HTTP errors
			data = response.json()

			self.raw_zeros_t = sorted([Decimal(item['height_t_str']) for item in data if 'height_t_str' in item])
			print(f"Fetched {len(self.raw_zeros_t)} zeros from API.")
			if not self.raw_zeros_t:
				print(f"Warning: No zeros fetched. Check API or parameters. Response: {data}")

		except requests.exceptions.RequestException as e:
			print(f"Error fetching zeros from API: {e}")
		except Exception as e:
			print(f"Error processing zero data: {e}")


	def _calculate_spacings(self) -> List[float]:
		"""Calculates differences between consecutive zeros."""
		if len(self.raw_zeros_t) < 2:
			return []
		spacings = [float(self.raw_zeros_t[i+1] - self.raw_zeros_t[i]) for i in range(len(self.raw_zeros_t) - 1)]
		return spacings

	def _normalize_spacings(self, spacings: List[float]) -> List[float]:
		"""
		Normalizes spacings, e.g., by dividing by the average spacing or using log(T/2pi) factor.
		This is a crucial step and depends on the specific model of zero statistics (e.g., GUE implies mean spacing of 1 after normalization).
		A common normalization for zeros t_n is g_n = (t_{n+1} - t_n) * log(t_n / 2pi) / (2pi)
		For simplicity here, we'll just use raw spacings or simple mean normalization.
		"""
		if not spacings:
			return []

		# Example: Simple mean normalization (not standard for zeta zeros)
		# mean_spacing = sum(spacings) / len(spacings)
		# normalized = [s / mean_spacing for s in spacings]
		# return normalized

		# Placeholder: Use raw spacings for now. Proper normalization is key.
		print("Warning: Using raw spacings. Proper normalization is required for meaningful results.")
		return spacings

	def get_sequences(self) -> Optional[np.ndarray]:
		"""
		Generates sequences of normalized spacings.
		Each sequence will be of length `self.sequence_length`.
		The target for each sequence is the next spacing.
		Returns:
			A NumPy array of shape (num_sequences, sequence_length) for input
			and (num_sequences,) for target, or None if not enough data.
		"""
		spacings = self._calculate_spacings()
		if not spacings:
			print("No spacings calculated, cannot generate sequences.")
			return None

		normalized_spacings = self._normalize_spacings(spacings)
		if len(normalized_spacings) < self.sequence_length + 1:
			print(f"Not enough normalized spacings ({len(normalized_spacings)}) to form a sequence of length {self.sequence_length + 1}.")
			return None

		sequences = []
		targets = []
		for i in range(len(normalized_spacings) - self.sequence_length):
			sequences.append(normalized_spacings[i : i + self.sequence_length])
			targets.append(normalized_spacings[i + self.sequence_length])

		if not sequences:
			print("No sequences generated.")
			return None

		return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)


# --- Transformer Model Definition (using Flax) ---

class MultiHeadAttention(nn.Module):
	num_heads: int
	head_dim: int
	dropout_rate: float = 0.1

	@nn.compact
	def __call__(self, inputs_q, inputs_kv, mask=None, deterministic=True):
		q_proj = nn.Dense(self.num_heads * self.head_dim, kernel_init=nn.initializers.xavier_uniform())(inputs_q)
		k_proj = nn.Dense(self.num_heads * self.head_dim, kernel_init=nn.initializers.xavier_uniform())(inputs_kv)
		v_proj = nn.Dense(self.num_heads * self.head_dim, kernel_init=nn.initializers.xavier_uniform())(inputs_kv)

		def split_heads(x):
			return x.reshape(x.shape[:-1] + (self.num_heads, self.head_dim)).transpose((0, 2, 1, 3))

		q, k, v = map(split_heads, (q_proj, k_proj, v_proj))

		attention_logits = nn.dot_product_attention_weights(q, k, bias=mask, deterministic=deterministic, dropout_rate=self.dropout_rate)
		attention_output = jnp.einsum('...hqk,...khd->...qhd', attention_logits, v)

		attention_output = attention_output.transpose((0, 2, 1, 3)).reshape(inputs_q.shape[:-1] + (self.num_heads * self.head_dim))
		output = nn.Dense(inputs_q.shape[-1], kernel_init=nn.initializers.xavier_uniform())(attention_output)
		return output

class PositionWiseFFN(nn.Module):
	hidden_dim: int
	output_dim: int # Should match input embedding dim for residual connection
	dropout_rate: float = 0.1

	@nn.compact
	def __call__(self, x, deterministic=True):
		y = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.relu_kernel_initializer())(x)
		y = nn.relu(y)
		y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)
		y = nn.Dense(self.output_dim, kernel_init=nn.initializers.xavier_uniform())(y)
		return y

class TransformerEncoderLayer(nn.Module):
	embed_dim: int # Dimension of input embeddings
	num_heads: int
	ffn_hidden_dim: int
	dropout_rate: float = 0.1

	@nn.compact
	def __call__(self, x, mask=None, deterministic=True):
		# Self-attention
		attn_output = MultiHeadAttention(num_heads=self.num_heads, head_dim=self.embed_dim // self.num_heads, dropout_rate=self.dropout_rate)(
			x, x, mask=mask, deterministic=deterministic
		)
		attn_output = nn.Dropout(rate=self.dropout_rate)(attn_output, deterministic=deterministic)
		x = nn.LayerNorm()(x + attn_output)

		# Position-wise FFN
		ffn_output = PositionWiseFFN(hidden_dim=self.ffn_hidden_dim, output_dim=self.embed_dim, dropout_rate=self.dropout_rate)(
			x, deterministic=deterministic
		)
		ffn_output = nn.Dropout(rate=self.dropout_rate)(ffn_output, deterministic=deterministic)
		x = nn.LayerNorm()(x + ffn_output)
		return x

class ZetaSpacingTransformer(nn.Module):
	num_layers: int
	embed_dim: int
	num_heads: int
	ffn_hidden_dim: int
	max_len: int # Max sequence length for positional encoding
	dropout_rate: float = 0.1
	# Output could be a single value (next spacing) or parameters of a distribution
	output_dim: int = 1 # Predicting the next spacing directly

	def setup(self):
		# Input embedding for spacings (since they are continuous, this acts like a projection)
		self.input_projection = nn.Dense(self.embed_dim)

		# Positional encoding
		pos_enc_shape = (1, self.max_len, self.embed_dim)
		# Standard sinusoidal positional encoding
		position = jnp.arange(0, self.max_len, dtype=jnp.float32)[:, None]
		div_term = jnp.exp(jnp.arange(0, self.embed_dim, 2) * -(jnp.log(10000.0) / self.embed_dim))
		pe = jnp.zeros(pos_enc_shape)
		pe = pe.at[:, :, 0::2].set(jnp.sin(position * div_term))
		pe = pe.at[:, :, 1::2].set(jnp.cos(position * div_term))
		self.positional_encoding = pe

		self.encoder_layers = [
			TransformerEncoderLayer(
				embed_dim=self.embed_dim,
				num_heads=self.num_heads,
				ffn_hidden_dim=self.ffn_hidden_dim,
				dropout_rate=self.dropout_rate
			) for _ in range(self.num_layers)
		]
		self.output_projection = nn.Dense(self.output_dim) # Predict the next spacing

	def __call__(self, x, deterministic=True, attention_mask=None):
		# x shape: (batch_size, sequence_length) - sequence of spacings
		# Project input spacings to embedding dimension
		# Input x is (batch_size, seq_len), needs to be (batch_size, seq_len, 1) for Dense
		x_projected = self.input_projection(x[..., None]) # (batch_size, seq_len, embed_dim)

		# Add positional encoding
		seq_len = x_projected.shape[1]
		x_projected = x_projected + self.positional_encoding[:, :seq_len, :]
		x_projected = nn.Dropout(rate=self.dropout_rate)(x_projected, deterministic=deterministic)

		# Create causal mask if not provided (for autoregressive prediction)
		if attention_mask is None:
			attention_mask = nn.make_causal_mask(x_projected[:, :, 0]) # Mask based on seq_len

		# Pass through Transformer encoder layers
		encoded = x_projected
		for layer in self.encoder_layers:
			encoded = layer(encoded, mask=attention_mask, deterministic=deterministic)

		# Use the output of the last token in the sequence to predict the next spacing
		# Or, some models use global average pooling over sequence outputs
		last_token_representation = encoded[:, -1, :] # (batch_size, embed_dim)

		# Project to output dimension (e.g., 1 for direct prediction of next spacing)
		next_spacing_prediction = self.output_projection(last_token_representation)
		return next_spacing_prediction # (batch_size, output_dim)

# --- Training Loop ---

@jax.jit
def train_step(params, opt_state, batch_inputs, batch_targets, model_apply_fn, optimizer_update_fn, key):
	"""Performs a single training step."""

	def loss_fn(params_):
		# `deterministic=False` for training (enables dropout)
		# `rngs={'dropout': key}` for dropout randomness
		predictions = model_apply_fn({'params': params_}, batch_inputs, deterministic=False, rngs={'dropout': key})
		# Assuming output_dim=1, predictions shape (batch_size, 1), targets (batch_size,)
		loss = jnp.mean((predictions.squeeze() - batch_targets) ** 2) # Mean Squared Error
		return loss

	grad_fn = jax.grad(loss_fn)
	grads = grad_fn(params)

	updates, new_opt_state = optimizer_update_fn(grads, opt_state, params)
	new_params = optax.apply_updates(params, updates)

	loss_val = loss_fn(params) # Calculate loss with old params for logging
	return new_params, new_opt_state, loss_val

def train_model(dataset: ZetaZeroDataset, model: ZetaSpacingTransformer, key: jax.random.PRNGKey):
	"""Main training loop."""

	sequences_data = dataset.get_sequences()
	if sequences_data is None:
		print("Failed to get sequences from dataset. Aborting training.")
		return None

	train_inputs, train_targets = sequences_data

	if train_inputs.shape[0] < DEFAULT_BATCH_SIZE:
		print(f"Not enough data ({train_inputs.shape[0]} samples) for batch size {DEFAULT_BATCH_SIZE}. Try fetching more data or reducing batch size.")
		return None

	# Initialize model parameters and optimizer state
	key_init, key_dropout, key_train_loop = jax.random.split(key, 3)

	# Dummy input for initialization
	dummy_input = jnp.ones((DEFAULT_BATCH_SIZE, dataset.sequence_length), dtype=jnp.float32)
	params = model.init(key_init, dummy_input, deterministic=True)['params'] # Pass dummy input

	# Optimizer (e.g., Adam)
	# You can also add a learning rate scheduler here
	tx = optax.adam(learning_rate=LEARNING_RATE)
	opt_state = tx.init(params)

	num_batches = train_inputs.shape[0] // DEFAULT_BATCH_SIZE

	print(f"Starting training for {NUM_EPOCHS} epochs...")
	for epoch in range(NUM_EPOCHS):
		key_dropout, key_epoch_shuffle = jax.random.split(key_dropout)

		# Shuffle data (optional, good practice)
		perm = jax.random.permutation(key_epoch_shuffle, train_inputs.shape[0])
		shuffled_inputs = train_inputs[perm]
		shuffled_targets = train_targets[perm]

		epoch_loss = 0.0
		for batch_idx in range(num_batches):
			key_train_loop, _ = jax.random.split(key_train_loop) # New key for each step for dropout

			start_idx = batch_idx * DEFAULT_BATCH_SIZE
			end_idx = start_idx + DEFAULT_BATCH_SIZE

			batch_inputs = shuffled_inputs[start_idx:end_idx]
			batch_targets = shuffled_targets[start_idx:end_idx]

			params, opt_state, loss_val = train_step(
				params, opt_state, batch_inputs, batch_targets,
				model.apply, tx.update, key_train_loop
			)
			epoch_loss += loss_val

		avg_epoch_loss = epoch_loss / num_batches
		print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Loss: {avg_epoch_loss:.6f}")

	print("Training finished.")
	return params # Return trained parameters

# --- Main Execution ---
def main():
	print("Phase 2: Signal Analysis - Transformer Training for Zeta Zeros")

	# Initialize dataset
	dataset = ZetaZeroDataset(api_url=PHASE1_API_URL, sequence_length=DEFAULT_SEQUENCE_LENGTH)

	# Fetch some initial data
	# These ranges should ideally cover enough zeros to generate many sequences.
	# For a real run, you might fetch zeros in segments or use a much larger range.
	dataset.fetch_zeros(start_height_t="0", end_height_t="2000", limit=2000) # Fetch more zeros

	if not dataset.raw_zeros_t or len(dataset.raw_zeros_t) < dataset.sequence_length + 1:
		print("Not enough raw zeros fetched to proceed with training. Exiting.")
		return

	# Initialize PRNG key for JAX
	key = jax.random.PRNGKey(0)

	# Instantiate the model
	# These hyperparameters are examples and should be tuned.
	model = ZetaSpacingTransformer(
		num_layers=3,
		embed_dim=64,
		num_heads=4,
		ffn_hidden_dim=128,
		max_len=dataset.sequence_length, # Or a bit larger if sequences can vary
		dropout_rate=0.1
	)

	# Train the model
	trained_params = train_model(dataset, model, key)

	if trained_params:
		print("Model training complete. Trained parameters obtained.")
		# Here you would save the trained_params, e.g., using Flax serialization
		# from flax.training import checkpoints
		# checkpoints.save_checkpoint(ckpt_dir='./model_checkpoints', target=trained_params, step=NUM_EPOCHS)
		print("Next steps: Evaluate the model, use it for predictions, or analyze learned attention patterns.")
	else:
		print("Model training failed or was aborted.")

if __name__ == "__main__":
	main()

# Requirements:
# pip install jax jaxlib flax optax requests numpy
#
# Ensure Phase 1 API (data_verse/api/main.py) is running and accessible at PHASE1_API_URL.
# The API should serve zero data, which this script consumes.
#
# Important Considerations:
# 1. Normalization: The `_normalize_spacings` method is critical.
#	Standard GUE normalization for zeta zeros involves log(T/2pi) factor.
#	This ensures that statistically, the mean spacing is 1.
# 2. Output Layer: The current model predicts the next spacing directly (regression).
#	Alternatively, it could predict parameters of a distribution (e.g., mean and variance
#	if modeling the next spacing as a Gaussian, or parameters of a GUE-like distribution).
# 3. Data Scale: The number of zeros required for robust training can be very large.
#	The `fetch_zeros` limit and ranges in `main()` are small for demonstration.
# 4. Evaluation: A proper evaluation loop (on a held-out test set) is needed to assess model performance.
#	Metrics could include Mean Squared Error, or more sophisticated measures related to point process modeling.
# 5. Causal Masking: The `make_causal_mask` is important for autoregressive prediction, ensuring
#	the model only attends to past positions when predicting the next.
# 6. Hyperparameter Tuning: All model dimensions, learning rate, batch size, etc., are hyperparameters
#	that would need careful tuning.
```

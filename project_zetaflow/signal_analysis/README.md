# Phase 2: Signal Analysis (Generative Zero Model)

**Objective:** To move beyond simple pair-correlation statistics (like the Montgomery-Odlyzko law, which relates zero spacings to Random Matrix Theory's GUE model) and build a deep learning model that captures the complex, higher-order correlations and long-range dependencies in the sequence of Riemann zeta function zeros.

The core idea is to treat the sequence of normalized zeros as a mathematical "language." By training a powerful sequence model, such as a Transformer, we aim to force it to learn the underlying "grammar" or statistical structure that governs the distribution of these zeros.

## Components

1.  **`train_transformer.py`**: A Python script using JAX with Flax/Equinox (or PyTorch) to:
    *   **Data Loading (`ZetaZeroDataset`)**: Connects to the Phase 1 API (`data_verse`), fetches sequences of zeros, calculates their normalized spacings, and prepares them as input tensors for the model.
        *   **Normalization:** This is a critical step. For the Riemann zeros $1/2 + i \gamma_n$, a common normalization for the spacings $\delta_n = \gamma_{n+1} - \gamma_n$ is $\tilde{\delta}_n = \delta_n \frac{\log(\gamma_n/2\pi)}{2\pi}$. Under the GUE hypothesis, these normalized spacings $\tilde{\delta}_n$ should have a mean of 1.
    *   **Sequence Model (`ZetaSpacingTransformer`)**: Defines a Transformer-based architecture. The model takes a sequence of (normalized) zero spacings as input and is trained to predict the distribution or value of the next zero spacing.
        *   **Causal Masking:** The Transformer uses causal attention masks to ensure that the prediction for a spacing at a position only depends on the preceding spacings, making it autoregressive.
    *   **Training Loop**: Implements the training process, including:
        *   Loss function (e.g., Mean Squared Error if predicting the next spacing directly, or a likelihood loss if predicting a distribution).
        *   Optimizer (e.g., Adam, AdamW).
        *   Learning rate scheduling.
        *   Parameter initialization and updates.

## Technology Stack

*   **Frameworks:** JAX with Flax/Equinox is preferred for its performance, scalability, and strong support for research. PyTorch is a viable alternative.
*   **Model Architecture:** Transformer.
*   **Data Source:** Phase 1 API (`data_verse`).

## Getting Started

### Prerequisites

*   Python 3.8+
*   JAX, Flax, Optax: `pip install jax jaxlib flax optax`
*   Requests: `pip install requests`
*   NumPy: `pip install numpy`
*   Ensure the Phase 1 Data-Verse API is running and accessible at the URL specified in `train_transformer.py` (default: `http://localhost:8000`).

### Running the Training Script

1.  **Configure `PHASE1_API_URL`**: If your Data-Verse API is running on a different host or port, update this constant at the beginning of `train_transformer.py`.
2.  **Fetch Sufficient Data**: The `main()` function in `train_transformer.py` includes a call to `dataset.fetch_zeros(...)`. Ensure the `start_height_t`, `end_height_t`, and `limit` parameters are set to fetch a substantial number of zeros. Training a Transformer effectively requires a large dataset.
3.  **Execute the script**:
    ```bash
    python train_transformer.py
    ```

The script will:
1.  Initialize the `ZetaZeroDataset`.
2.  Fetch zero data from the Phase 1 API.
3.  Preprocess the data into sequences of normalized spacings.
4.  Initialize the `ZetaSpacingTransformer` model.
5.  Run the training loop for the specified number of epochs.
6.  Print training progress (average loss per epoch).

## Model Purpose and Interpretation

The goal of this phase is not just to predict the next zero with high accuracy (though that would be an indicator of success), but to learn the *deep statistical structure* of the zero sequence. This involves:

*   **Higher-Order Correlations:** Moving beyond pair correlations (2-point functions) to understand n-point correlation functions.
*   **Long-Range Dependencies:** Capturing how zeros far apart in the sequence might still influence each other in subtle ways not explained by local statistics.
*   **"Grammar" of Zeros:** If the zeros indeed behave like a language, this model attempts to learn its rules.
*   **Attention Analysis:** The learned attention patterns within the Transformer can be analyzed to understand which past zeros (or spacings) the model deems important when predicting the next one. This could provide new mathematical insights or highlight interesting structures.

The output of this phase will be:
*   A trained Transformer model (its parameters).
*   Potentially, new statistical insights or conjectures about the distribution of zeros derived from analyzing the model's behavior and performance.

This learned structure can then inform the design of the "analytic probe" in Phase 4, for instance, by guiding the architecture or initialization of the neural network representing the probe.

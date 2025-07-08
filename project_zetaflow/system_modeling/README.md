# Phase 3: System Modeling (Differentiable Zeta Surrogate)

**Objective:** To create a fast, accurate, and differentiable software model (a "surrogate") of the Riemann zeta function, $\zeta(s)$, particularly in and around the critical strip ($0 < \text{Re}(s) < 1$). This surrogate is essential for gradient-based optimization in Phase 4 (The Discovery Engine).

The core approach is to use a Physics-Informed Neural Network (PINN). A PINN is a neural network trained to satisfy two types of conditions:
1.  **Data-driven constraints:** Matching known values of $\zeta(s)$ at various points $s$.
2.  **Physics-informed (or mathematics-informed) constraints:** Adhering to known mathematical properties of $\zeta(s)$, primarily the Riemann functional equation.

## Components

1.  **`train_pinn.py`**: A Python script using JAX with Flax/Equinox to:
    *   **Model Definition (`ZetaPINN`, `ComplexMLP`)**: Defines a neural network architecture capable of taking a complex number $s = \sigma + it$ as input and outputting a complex value approximating $\zeta(s)$. The architecture must handle complex arithmetic, either natively or by operating on real and imaginary parts.
    *   **Data Generation**:
        *   Generates a set of complex points $s_{data}$ and computes the corresponding true $\zeta(s_{data})$ values using a high-precision library like `mpmath`. These form the data loss component.
        *   Generates a set of collocation points $s_{collocation}$ where the functional equation will be enforced.
    *   **Loss Function (`total_loss_fn`)**: Implements a composite loss function:
        *   **Data Loss**: Measures the discrepancy between the PINN's output $\hat{\zeta}(s_{data})$ and the true values $\zeta(s_{data})$. Typically Mean Squared Error (MSE) on the complex values: $|\hat{\zeta}(s_{data}) - \zeta(s_{data})|^2$.
        *   **Physics Loss (Functional Equation Residual)**: Measures how well the PINN's output satisfies the Riemann functional equation: $\zeta(s) = \chi(s) \zeta(1-s)$. The residual is $|\hat{\zeta}(s_{col}) - \chi(s_{col}) \hat{\zeta}(1-s_{col})|^2$.
            *   The function $\chi(s) = 2^s \pi^{s-1} \sin(\frac{\pi s}{2}) \Gamma(1-s)$ must be implemented, ideally using JAX-compatible functions for automatic differentiation (e.g., `jax.scipy.special.gamma`).
    *   **Training Loop**: Implements the training process:
        *   Initializes model parameters.
        *   Uses an optimizer (e.g., Adam) to minimize the total loss function with respect to the model parameters.
        *   JAX's automatic differentiation capabilities are crucial here, especially for complex-valued functions and the $\Gamma$ function if its JAX version is used.

## Technology Stack

*   **Frameworks:** JAX with Flax/Equinox is highly preferred due to its strong support for automatic differentiation (including complex numbers), compilation (XLA), and suitability for scientific machine learning and PINNs.
*   **High-Precision Reference:** Python's `mpmath` library is used to generate accurate $\zeta(s)$ values for the data loss component.
*   **Architecture:** Physics-Informed Neural Network (PINN).

## Getting Started

### Prerequisites

*   Python 3.8+
*   JAX, Flax, Optax: `pip install jax jaxlib flax optax`
*   mpmath: `pip install mpmath`
*   NumPy: `pip install numpy`
*   SciPy: `pip install scipy` (for `jax.scipy.special.gamma`)

### Running the Training Script

1.  **Review Configuration**: Check constants at the beginning of `train_pinn.py` like `NUM_EPOCHS`, `LEARNING_RATE`, `BATCH_SIZE`, `NUM_DATA_POINTS`, `NUM_COLLOCATION_POINTS`, and `PHYSICS_LOSS_WEIGHT`. These may need tuning.
2.  **Execute the script**:
    ```bash
    python train_pinn.py
    ```

The script will:
1.  Generate training data (points $s_{data}$ with $\zeta(s_{data})$ values from `mpmath`, and collocation points $s_{collocation}$).
2.  Initialize the `ZetaPINN` model.
3.  Run the training loop, minimizing the combined data and physics loss.
4.  Print training progress (average total loss per epoch).
5.  After training, it will (optionally) save the trained model parameters and test the model on a few sample points, comparing its output to `mpmath` values.

## Model Purpose and Interpretation

The primary output of this phase is a **trained `ZetaPINN` model** whose parameters represent a differentiable surrogate for the Riemann zeta function. This surrogate should:

*   **Approximate $\zeta(s)$:** Provide reasonably accurate values for $\zeta(s)$ within its trained domain.
*   **Respect Functional Equation:** Approximately satisfy the Riemann functional equation due to the physics-informed loss term.
*   **Be Differentiable:** Allow gradients of $\zeta(s)$ (or rather, the surrogate $\hat{\zeta}(s)$) with respect to $s$ (both real and imaginary parts) to be computed efficiently using JAX's automatic differentiation. This is critical for Phase 4.

The quality of this surrogate (accuracy, adherence to functional equation, smoothness of its derivatives) will directly impact the effectiveness of the Discovery Engine in Phase 4. A well-trained PINN provides a computational "sandbox" where properties related to $\zeta(s)$ can be explored via optimization.

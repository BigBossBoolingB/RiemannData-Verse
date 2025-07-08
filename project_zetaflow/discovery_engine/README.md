# Phase 4: The Discovery Engine (Automated Probe Design)

**Objective:** To computationally discover an optimal "analytic probe" function $M(s)$ (a next-generation mollifier) and use it to generate a precise, testable mathematical conjecture that, if proven, could imply the Riemann Hypothesis (RH).

This phase leverages the outputs of previous phases:
*   **Phase 2 (Signal Analysis):** The trained Transformer model may provide insights or features to guide the architecture or initialization of the probe network.
*   **Phase 3 (System Modeling):** The trained `ZetaPINN` provides a differentiable surrogate for $\zeta(s)$, which is essential for the gradient-based optimization of the probe.

## Key Components & Functionality

1.  **`design_probe.py`**: A Python script (JAX/PyTorch) that:
    *   **Loads Models:** Loads the trained `ZetaPINN` (from Phase 3) and optionally the trained Transformer (from Phase 2).
    *   **Probe Network (`ProbeNet`)**: Defines a neural network $M(s)$ that represents the analytic probe function.
        *   The input to `ProbeNet` is a complex number $s$.
        *   The output is a complex value $M(s)$.
        *   Its architecture might be informed by statistical patterns learned by the Transformer (e.g., complexity, number of parameters, specific basis functions if the Transformer's output can be interpreted that way).
    *   **Inverse Problem Formulation**: Frames the search for $M(s)$ as an optimization problem. The goal is to find an $M(s)$ that minimizes a chosen objective function. A common objective for mollifiers is to make the product $M(s)\zeta(s)$ "close to 1" or have other desirable properties (like small variance) on the critical line.
        *   **Objective Function Example:** Minimize $\int_{\text{critical line}} |1 - M(s)\zeta_{\text{PINN}}(s)|^2 ds$. The integral is typically approximated by a sum over discrete points on the critical line $s = 1/2 + it$.
    *   **Optimization Loop**:
        *   Initializes the parameters of `ProbeNet`.
        *   Uses gradient-based optimization (e.g., Adam) to update the `ProbeNet` parameters by minimizing the objective function. This relies on JAX's (or PyTorch's) automatic differentiation capabilities to compute gradients through both `ProbeNet` and `ZetaPINN`.
    *   **Conjecture Extraction Support**:
        *   The primary output is the set of learned weights/parameters for `ProbeNet`.
        *   The script should also provide a summary or analysis of the learned $M(s)$ and the behavior of $M(s)\zeta_{\text{PINN}}(s)$. This is not a direct symbolic formula but numerical evidence and properties.
        *   This numerical evidence then needs to be interpreted by mathematicians to formulate a precise, symbolic conjecture.

## Technology Stack

*   **Frameworks:** JAX (with Flax/Equinox) or PyTorch, consistent with Phases 2 and 3. JAX is particularly well-suited due to its strong AD capabilities.
*   **Models:** Utilizes the trained `ZetaPINN` and potentially the `ZetaSpacingTransformer`.

## Getting Started

### Prerequisites

*   Python 3.8+
*   JAX, Flax, Optax: `pip install jax jaxlib flax optax`
*   NumPy: `pip install numpy`
*   Ensure that trained model parameters (checkpoints) for `ZetaPINN` (from Phase 3) and optionally `ZetaSpacingTransformer` (from Phase 2) are available and paths are correctly specified in `design_probe.py`. (The script currently uses placeholders/dummy loaders).

### Running the Script

1.  **Configure Model Paths**: Update `PATH_TO_PINN_CHECKPOINT` and `PATH_TO_TRANSFORMER_CHECKPOINT` in `design_probe.py` to point to your actual saved model parameters.
2.  **Review Optimization Parameters**: Check constants like `NUM_EPOCHS_PROBE`, `LEARNING_RATE_PROBE`, `BATCH_SIZE_PROBE`, `NUM_INTEGRATION_POINTS`, `INTEGRATION_T_MIN`, `INTEGRATION_T_MAX`.
3.  **Execute the script**:
    ```bash
    python design_probe.py
    ```

The script will:
1.  Attempt to load the `ZetaPINN` parameters and (if configured) the Transformer parameters.
2.  Initialize the `ProbeNet` model.
3.  Run the optimization loop, adjusting `ProbeNet`'s parameters to minimize the objective function.
4.  Print the optimization progress (average loss per epoch).
5.  Output information about the learned `ProbeNet` and a conceptual summary of a potential conjecture.

## Output and Interpretation

The direct output of this phase is the **learned `ProbeNet` model (its parameters)**.

The more challenging part is **conjecture extraction**. A neural network itself is not a symbolic mathematical formula. The process involves:
1.  **Numerical Analysis:** Analyzing the behavior of the learned $M(s)$ and the product $M(s)\zeta_{\text{PINN}}(s)$ across various regions of the complex plane, especially on and near the critical line.
    *   How close is $M(s)\zeta_{\text{PINN}}(s)$ to 1?
    *   What are the analytic properties of the learned $M(s)$ (e.g., poles, zeros, smoothness)?
    *   Does $M(s)$ resemble any known classes of functions used in analytic number theory?
2.  **Human Interpretation:** Mathematicians must interpret these numerical results and observed properties to formulate a precise, symbolic mathematical conjecture. This conjecture might state, for example:
    *   "There exists a function $M(s)$ with specific analytic properties (e.g., derived from the `ProbeNet` architecture or observed behavior) such that $\int |1 - M(s)\zeta(s)|^2 ds < \epsilon$ on the critical line, which implies RH."
    *   Or, it might lead to a conjecture about the properties of $\zeta(s)$ itself, made evident by the probe.

This phase aims to use AI as a powerful tool for *discovering patterns and relationships* that can guide human mathematicians towards new, provable conjectures. The "human-readable summary" from the script is a starting point for this deeper mathematical analysis. The conjecture is then passed to Phase 5 for formal verification.

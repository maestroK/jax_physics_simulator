# Neural Optics DARTS: Continuous Topology Search & Implicit Neural Representations for Wave Physics

This repository contains the official JAX implementation for automated discovery of super-resolution optical topologies. We formulate the design of physical optical experiments as a hybrid discrete-continuous optimization problem and solve it using Differentiable Architecture Search (DARTS) coupled with Implicit Neural Representations (INRs).

Unlike standard pixel-wise phase retrieval (which suffers from mode collapse and high-frequency speckle), this architecture forces the phase masks to map to physically realizable diffractive elements while simultaneously pruning redundant optical paths.

## Core Methodology

### 1. Differentiable Architecture Search (DARTS) for Optics
We relax the discrete topology of an optical table (e.g., beam splitters, mirrors) into a continuous super-graph. The routing of the complex scalar field $E(x, y)$ is parameterized by a continuous transmittance variable $T \in (0, 1)$:

$$E_{\text{transmit}} = \sqrt{T} E_{\text{in}}, \quad E_{\text{reflect}} = i\sqrt{1 - T} E_{\text{in}}$$

To enforce discretization at convergence, we apply an annealed entropy penalty to the routing logits:
$$\mathcal{L}_{\text{ent}} = - T \log(T) - (1-T) \log(1-T)$$

### 2. Random Fourier Feature (RFF) Phase Parameterization
Direct optimization of $N \times N$ spatial light modulator (SLM) pixels yields unphysical artifacts. We parameterize the phase mask $\Phi(x, y)$ using an MLP operating on normalized spatial coordinates. To overcome the spectral bias of coordinate networks and avoid axis-aligned grid artifacts, we inject Random Fourier Features (RFF) using a Gaussian projection matrix $\mathbf{B}$:

$$\gamma(\mathbf{v}) = [\sin(2\pi \mathbf{v} \mathbf{B}), \cos(2\pi \mathbf{v} \mathbf{B})]$$
$$\Phi_\theta(x, y) = \text{MLP}_\theta(\gamma(x, y)) \cdot \pi$$

### 3. Exact Wave Propagation
Free-space propagation is modeled via the exact Rayleigh-Sommerfeld (RS) transfer function, implemented as a fully compiled, JAX-accelerated spectral filter:

$$H(f_x, f_y, z) = \exp\left(i k z \sqrt{1 - (\lambda f_x)^2 - (\lambda f_y)^2}\right)$$

## Installation

The environment requires JAX, Flax, and Optax. 

```bash
# Standard CPU/GPU installation
pip install --upgrade pip
pip install jax[cuda12] flax optax matplotlib

# For Apple Silicon (M1/M2/M3) local testing
pip install jax-metal flax optax matplotlib

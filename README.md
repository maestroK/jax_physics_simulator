# Neural Optics DARTS: Differentiable Topology Search for Wave Physics

[![JAX](https://img.shields.io/badge/JAX-Accelerated-FF8800.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Neural Optics DARTS** bridges modern machine learning architecture search with exact wave physics. This repository provides a fully differentiable, JAX-accelerated framework for discovering complex, multi-path optical topologies (e.g., super-resolution microscopy setups) entirely from gradient descent.

By relaxing discrete optical graphs into continuous super-graphs and parameterizing phase elements with Implicit Neural Representations (INRs), we overcome the two fatal flaws of standard computational optics: topological rigidity and high-frequency speckle mode-collapse.

## 🚀 Key Innovations

### 1. Continuous Topology Search (DARTS)
We frame optical table design as a Differentiable Architecture Search (DARTS) problem. Beam splitters are parameterized by a continuous routing logit $T \in (0, 1)$, allowing gradients to flow through multi-path interferometers. An annealed binary entropy penalty forces the network to prune redundant paths and converge to a discrete, physically constructible topology.

### 2. Physics-Informed INRs via Random Fourier Features (RFF)
Direct optimization of $N \times N$ phase pixels exploits numerical grids, resulting in unphysical, high-total-variation (TV) speckle. We parameterize spatial light modulators (SLMs) using a coordinate-based MLP. To break the spectral bias and axis-aligned artifacts of standard coordinate networks, we inject **Random Fourier Features (RFF)**, forcing the network to discover smooth, manufacturable diffractive elements (yielding a **>70% reduction in phase TV** compared to baseline phase retrieval).

### 3. Batched Multi-Task Optimization (`jax.vmap`)
The architecture leverages JAX's `vmap` to concurrently evaluate the optical super-graph across multiple experimental targets (e.g., dynamically routing light to form a Gaussian focus in Path A and a Vortex beam in Path B). This achieves 100% GPU utilization during the discovery phase.

## ⚙️ Architecture & Physics Engine

* **Forward Model:** Exact scalar diffraction using a fully compiled, soft-bandlimited Rayleigh-Sommerfeld (RS) transfer function.
* **Loss formulation:** Approximated Strehl ratio optimization coupled with energy conservation penalties and DARTS entropy scheduling.
* **Hardware:** Native CPU/GPU scaling via XLA compilation.

## 📦 Installation

This framework requires JAX, Flax, and Optax.

```bash
# Standard GPU/CUDA installation
pip install --upgrade pip
pip install jax[cuda12] flax optax matplotlib

# For Apple Silicon (M1/M2/M3) local testing
pip install jax-metal flax optax matplotlib

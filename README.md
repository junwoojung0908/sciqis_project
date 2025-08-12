# Projects for *Scientific Computing in Quantum Information Science* (Course 10387, DTU)

A collection of small projects and a final project (TBD) completed during the **SCIQIS** course at the Technical University of Denmark (DTU).  
Each notebook focuses on a practical, computation-first view of quantum information topics.

---

## Table of Contents
- [Small Projects](#small-projects)
  - [basic_quantum_circuits.ipynb](#basic_quantum_circuitsipynb)
  - [gaussian_states_visualization.ipynb](#gaussian_states_visualizationipynb)
  - [rydberg_atom_annealing.ipynb](#rydberg_atom_annealingipynb)
  - [state_gate_classe.py](#state_gate_classepy)
- [Final Project](#final-project)
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)
- [Notes](#notes)

---

## Small Projects

### `basic_quantum_circuits.ipynb`
Demonstrates foundational circuits implemented in NumPy, including:
- **Quantum Teleportation**
- **Quantum Fourier Transform (QFT)** for three qubits

Focus: state preparation, simple gates, and step-by-step circuit logic.

---

### `gaussian_states_visualization.ipynb`
Visualizes **Gaussian operations** in continuous variables (CV) using **Wigner functions**.  
Includes:
- Contour and 3D surface plots of Gaussian states
- Displacement and squeezing demonstrations
- (Optional) timing utilities to profile plot generation

Focus: intuition for covariance matrices, displacement vectors, and how parameters map to phase‑space geometry.

---

### `rydberg_atom_annealing.ipynb`
Simulates quantum annealing on a **seven-qubit Rydberg atom graph**, tracking the time evolution of the full computational basis (up to 128 states).  
Plots relevant probabilities and provides a clear workflow for defining schedules and running time evolutions.

Focus: schedule design, time evolution, and probability tracking during annealing.

> **Note:** This notebook may rely on QuTiP for time evolution depending on the version used.

---

### `state_gate_classe.py`
Defines lightweight **`State`** and **`Gate`** abstractions (NumPy-based) along with basic operations (e.g., subsystem measurement).  
Companion module **`useful_states_gates.py`** includes representative states and gates (Pauli matrices, Bell states, etc.).

Focus: simple, readable building blocks for experimenting with circuits without a heavyweight framework.

---

## Final Project
**TBD** – work in progress. This section will include the final deliverables (notebook(s), report, and any scripts).

---

## Getting Started

1. **Clone or unzip** this repository locally.
2. (Recommended) Create a virtual environment:
   ```bash
   python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```
3. **Install dependencies** (see below). For a minimal setup:
   ```bash
   pip install numpy matplotlib jupyter
   ```
   If you plan to run the Rydberg annealing notebook:
   ```bash
   pip install qutip
   ```
4. **Launch Jupyter** and open the notebooks:
   ```bash
   jupyter notebook
   ```

---

## Dependencies
- **Core:** `numpy`, `matplotlib`, `jupyter`
- **Optional:** `qutip` (for time evolution in Rydberg annealing)

> Versions tested during the course varied by machine; recent Python 3.10+ should work fine.

---

## Notes
- Some notebooks include a lightweight `@timeit` decorator and `timed(...)` context manager to measure runtime and guide future optimizations.
- Plots aim to be self-explanatory: key parameters appear in titles/labels when useful.
- If you encounter environment or import issues, restart the kernel and execute cells from top to bottom.


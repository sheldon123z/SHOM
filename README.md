# SHOM: Sensitivity-based Heterogeneous Ordered Multi-agent Reinforcement Learning

[![IEEE TSG](https://img.shields.io/badge/IEEE%20TSG-2025-blue)](https://ieeexplore.ieee.org/document/10879343)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FTSG.2025.3540416-green)](https://doi.org/10.1109/TSG.2025.3540416)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Sensitivity-based Heterogeneous Ordered Multi-agent Reinforcement Learning for Distributed Volt-Var Control in Active Distribution Network**
>
> Published in **IEEE Transactions on Smart Grid**, February 2025

[Paper](https://ieeexplore.ieee.org/document/10879343) | [Project Page](https://xjtu-rl.github.io/SHOM/) | [PowerZoo](https://github.com/XJTU-RL/PowerZoo)

---

## Overview

<p align="center">
  <img src="docs/assets/images/CTDE_CHOM.png" alt="SHOM Framework" width="80%">
</p>

As power grids expand, maintaining stable voltage and minimizing losses become increasingly crucial. This paper proposes **SHOM** (Sensitivity-based Heterogeneous Ordered Multi-agent reinforcement learning), a novel approach for Volt-Var Control (VVC) in Active Distribution Networks (ADNs).

### Key Features

- **Heterogeneous Device Control**: Simultaneously controls regulators, capacitors, and batteries with different action spaces
- **Sensitivity-based Ordering**: Leverages V-Q sensitivity to determine optimal agent update sequences
- **CTDE Framework**: Centralized training with decentralized execution for robust distributed control
- **Monotonic Improvement**: Ensures consistent performance gains during policy updates

---

## Architecture

<p align="center">
  <img src="docs/assets/images/order_selector2.png" alt="Order Selector" width="70%">
</p>

The order selector determines agent update sequences based on voltage-reactive power (V-Q) sensitivity, enabling the model to capture temporal dynamics and adapt to system changes.

---

## Results

### Performance Comparison (123-Bus Network)

| Algorithm | Avg. Reward | Convergence Steps | Stability |
|-----------|-------------|-------------------|-----------|
| **SHOM**  | **-12.05**  | **~5,000**        | **Best**  |
| HAPPO     | -14.2       | ~8,000            | Good      |
| Qmix      | -13.8       | ~7,500            | Good      |
| PPO       | -16.5       | ~10,000           | Fair      |
| SAC       | -15.8       | ~9,000            | Fair      |

### Key Metrics (34-Bus Network)

- **Power Loss Reduction**: 66% lower than other algorithms (0.72 kW avg.)
- **Voltage Violations**: 0 violations under SHOM control
- **Device Utilization**: Optimal battery usage for grid regulation

---

## Citation

```bibtex
@article{zhengSensitivityBasedHeterogeneousOrdered2025,
  author  = {Xiaodong Zheng and Shixuan Yu and Hui Cao and Tianzhuo Shi and Shuangsi Xue and Tao Ding},
  title   = {Sensitivity-Based Heterogeneous Ordered Multi-Agent Reinforcement Learning for Distributed Volt-Var Control in Active Distribution Network},
  journal = {IEEE Transactions on Smart Grid},
  year    = {2025},
  month   = {Feb},
  doi     = {10.1109/TSG.2025.3540416},
  issn    = {1949-3061},
  url     = {https://ieeexplore.ieee.org/document/10879343}
}
```

---

## Code Implementation

The algorithm is implemented in **PowerZoo**, a Python-based reinforcement learning simulation environment for power systems.

```bash
git clone https://github.com/XJTU-RL/PowerZoo.git
cd PowerZoo
conda env create -f environment.yml
conda activate powerzoo
python examples/train.py
```

For more details, visit the [PowerZoo Repository](https://github.com/XJTU-RL/PowerZoo).

---

## Authors

**Xi'an Jiaotong University** - Shaanxi Key Laboratory of Smart Grid

- Xiaodong Zheng
- Shixuan Yu
- **Hui Cao** *(Corresponding Author)*
- Tianzhuo Shi
- Shuangsi Xue
- Tao Ding

---

## Acknowledgments

This work was supported by the **National Natural Science Foundation of China** (Grant 52277123).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

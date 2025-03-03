# PowerZoo


PowerZoo 是一个基于 python 和 opendss 的电力系统强化学习仿真环境。它旨在提供一个灵活且易于使用的平台，用于训练和评估强化学习算法在电力系统中的应用。


## 安装：


```
git clone https://github.com/XJTU-RL/PowerZoo.git
```

推荐使用 conda 环境（可以在`enviornment.yml`中修改环境名称）

```
cd PowerZoo
conda env create -f environment.yml
conda activate powerzoo
```

## 快速开始


```
python examples/train.py
```


## 环境变量


```
python powerzoo/main.py --help
```

Please cite the following paper if you use this repo for scientific research:

[1] X. Zheng, S. Yu, H. Cao, T. Shi, S. Xue, and T. Dingc, “Sensitivity-Based Heterogeneous Ordered Multi-Agent Reinforcement Learning for Distributed Volt-Var Control in Active Distribution Network,” IEEE Transactions on Smart Grid, pp. 1–1, Feb. 2025, doi: 10.1109/TSG.2025.3540416.
```
@article{zhengSensitivityBasedHeterogeneousOrdered2025,
  author  = {Xiaodong Zheng and Shixuan Yu and Hui Cao and Tianzhuo Shi and Shuangsi Xue and Tao Ding},
  title   = {Sensitivity-Based Heterogeneous Ordered Multi-Agent Reinforcement Learning for Distributed Volt-Var Control in Active Distribution Network},
  journal = {IEEE Transactions on Smart Grid},
  year    = {2025},
  month   = {Feb},
  doi     = {10.1109/TSG.2025.3540416},
  issn    = {1949-3061},
  url     = {https://ieeexplore.ieee.org/document/10879343},
  urldate = {2025-02-12}
}
```

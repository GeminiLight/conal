# Code of CONAL

This is the implementation of CONAL proposed in the paper "Towards Constraint-aware Learning for Resource Allocation in NFV-enabled Networks".


## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Algorithms](#algorithms)
  - [Datasets](#datasets)
  - [Experiment Scripts](#experiment-scripts)
- [Run Main Experiments](#run-experiments)
  - [Overall Evaluation and Ablation Study](#overall-evaluation-and-ablation-study)
  - [Generalizability Study](#generalizability-study)
    - [Request Frequency Sensitivity Study](#request-frequency-sensitivity-study)
    - [Dynamic Request Distribution Testing](#dynamic-request-distribution-testing)
  - [Scalability Analysis](#scalability-analysis)
    - [Large-scale Network Validation](#large-scale-network-validation)
    - [Solving Time Scale Analysis](#solving-time-scale-analysis)
  - [Real-world Network System Validation](#real-world-network-system-validation)
- [File Structure](#file-structure)
- [Acknowledgement](#acknowledgement)

## Installation

We recommend using CUDA devices to accelerate the training process:

```shell
# use cuda (optional version: 10.2, 11.3)
bash install.sh -c 12.1
```

If you do not have a device with GPUs, you can also use the CPU to run the code:

```shell
bash install.sh -c 0
```

## Quick Start

### Algorithms

In this library, we implement our proposed algorithms and baselines for the VNE problem.

| Algorithm | Type | Command `solver_name` | Brief Description |
| --- | --- | --- | --- |
| NRM-VNE | Node Ranking-based Heuristic | `nrm_rank` | Node Resource Management |
| GRC-VNE | Node Ranking-based Heuristic | `grc_rank` | Global Resource Control |
| NEA-VNE | Node Embedding-based Heuristic | `nea_rank` | Node Essentiality Assessment |
| PSO-VNE | Meta-heuristic | `pso_vne` | Particle Swarm Optimization |
| GA-VNE | Meta-heuristic | `ga_vne` | Genetic Algorithm | 
| MCTS-VNE | Model-based RL | `mcts` | Monte Carlo Tree Search |
| PG-CNN | Model-free RL | `pg_cnn` | Policy Gradient with Convolutional Neural Network |
| DDPG-Attention | Model-free RL | `ddpg_attention` | Deep Deterministic Policy Gradient with Attention |
| A3C-GCN | Model-free RL | `a3c_gcn` | Asynchronous Advantage Actor-Critic with Graph Convolutional Network |
| GAL-VNE | Model-free RL | Official Code | Global Reinforcement Learning and Local One-Shot Neural Prediction |


### Datasets

We evaluate these algorithms on the following four topologies:

| Topology | Command `topology` | Number of Nodes | Number of Links | Description |
| --- | --- | --- | --- | --- |
| WX100 | `wx100` | 100 | ~500 | A Waxman random graph with 100 nodes |
| WX500 | `wx500` | 500 | ~13,000 | A Waxman random graph with 500 nodes |
| Geant | `geant` | 40 | 64 | Europe’s national research and education networks |
| Brain | `brain` | 161 | 166 | Data network for scientific and cultural institutions in Berlin |

### Experiment Scripts

We provide shell scripts to run the experiments. You can modify the parameters in the shell scripts to run the experiments with different settings.

| Experiment | Shell Script | Description |
| --- | --- | --- |
| Overall Evaluation | `run_overall_performance.sh` | Run all algorithms on WX100 |
| Ablation Study | `run_ablation_study.sh` | Run CONAL and its variations on WX100 |
| Request Frequency Sensitivity Study | `run_request_frequency_sensitive_study.sh` | Run all algorithms on WX100 with different request frequencies |
| Dynamic Request Distribution Testing | `run_dynamic_request_distribution_testing.sh` | Run all algorithms on WX100 with dynamic request distribution |
| Scalability Validation | `run_scalability_validation.sh` | Run all algorithms on WX500 |
| Real-world Network System Validation | `run_real_world_network_system_validation.sh` | Run all algorithms on Geant and Brain |

### Run Main Experiments

To run the experiments, you run shell scripts by modifying several key parameters in the shell scripts.

| Parameter | Description | Options |
| --- | --- | --- |
| `topology` | Name of topology | [wx100, wx500, geant, brain] |
| `solver_name` | Name of solver | [nrm_rank, nea_rank, pso_vne, mcts, a3c_gcn, pg_cnn, ddpg_attention, conal] |
| `num_train_epochs` | Number of training epochs | [0, >0] |
| `use_pretrained_model` | Whether to use pretrained model | [1, 0] |
| `pretrained_model_path` | Path to the pretrained model | ['null', $path] |


### Overall Evaluation and Ablation Study

Specify the `solver_name` and run the following command:

```shell
bash run_overall_evaluation_and_ablation_study.sh
```

### Generalizability Study


#### Request Frequency Sensitivity Study

Specify the `solver_name` and run the following command:

```shell
bash run_generalizability_request_frequency_sensitive_study.sh
```

#### Dynamic Request Distribution Testing

Specify the `solver_name` and run the following command:

```shell
bash run_generalizability_dynamic_request_distribution_testing.sh
```

### Scalability Analysis


#### Large-scale Network Validation

Specify the `solver_name` and run the following command:

```shell
bash run_scalability_large_scale_network_validation.sh
```

#### Solving Time Scale Analysis

Specify the `solver_name` and run the following command:

```shell
bash run_scalability_solving_time_scale_analysis.sh
```

### Real-world Network System Validation

Specify the `solver_name` and `topology` and run the following command:

```shell
bash run_real_world_network_system_validation.sh
```

## File Structure

```shell
.
├── args.py
├── main.py
├── settings
│   ├── p_net_setting.yaml  # Simulation setting of physical network 
│   ├── v_sim_setting.yaml  # Simulation setting of virtual network request simulator 
└── virne
    ├── base                # Core components: environment, controller, recorder, scenario, solution
    ├── config.py           # Configuration class
    ├── data                # Data class: attribute, generator, network, physical_network, virtual_network, virtual_network_request_simulator
    ├── solver
    │   ├── heuristic                                    # 
    │   │   ├── node_rank.py                             # NRM-VNE, GRC-VNE, and NEA-VNE
    │   ├── learning                                     # 
    │   │   ├── a3c_gcn                                  # A3C-GCN
    │   │   ├── conal                                    # CONAL and its variations
    │   │   ├── ddpg_attention                           # DDPG-Attention
    │   │   ├── mcts                                     # MCTS
    │   │   └── pg_cnn                                   # PG-CNN
    │   ├── meta_heuristic                               #
    │   │   ├── genetic_algorithm_solver.py              # GA-VNE
    │   │   └── particle_swarm_optimization_solver.py    # PSO-VNE
    │   ├── registry.py
    │   └── solver.py
    └── utils
```


## Acknowledgement

- [Virne](https://github.com/GeminiLight/virne): It is a simulator for resource allocation problems in network virtualization, mainly for VNE problem.
- [SDNlib](https://github.com/GeminiLight/virne): It is a dataset for network simulation, mainly for network topology and network traffic generation.
- [GAL-VNE](https://github.com/Thinklab-SJTU/GAL-VNE): It is the implementation of paper "GAL-VNE: Solving the VNE Problem with Global Reinforcement Learning and Local One-Shot Neural Prediction"

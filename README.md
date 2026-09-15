# ACGRL: Adversarial Consistency-Guided Representation Learning for Multi-View Clustering

## Overview

ACGRL is a two-stage representation learning framework for multi-view clustering. It learns view-invariant semantics and view-specific complementary information in separate optimization stages, avoiding their direct joint optimization.

In Stage I, a Gradient Reversal Layer (GRL) and a view discriminator are used to suppress source-view information and learn consistent representations. In Stage II, the consistent representation network is frozen, and the learned invariant representations serve as semantic anchors for extracting view-specific residual information. A joint reconstruction objective preserves informative content, while cross-view cluster-assignment alignment produces clustering-friendly comprehensive representations.

This repository contains the clustering-only implementation of ACGRL. Classification, visualization, and auxiliary ablation code are not included.

## Supported Datasets

The project includes the following four multi-view datasets:

| Dataset | Samples | Views | Input dimensions | Clusters |
|---|---:|---:|---|---:|
| NGs | 500 | 3 | 2000 / 2000 / 2000 | 5 |
| BBCSport | 544 | 2 | 3183 / 3203 | 5 |
| Hdigit | 10,000 | 2 | 784 / 256 | 10 |
| Cora | 2,708 | 2 | 2708 / 1433 | 7 |

The corresponding `.mat` files are stored in the `data/` directory.

## Project Structure

```text
ACGRL/
├── data/
│   ├── BBCSport.mat
│   ├── Cora.mat
│   ├── Hdigit.mat
│   └── NGs.mat
├── units/
│   ├── ConfigFunction.py   # Dataset-specific model configurations
│   ├── ParserConfig.py     # Command-line arguments
│   ├── datasets.py         # Dataset loading and preprocessing
│   ├── loss.py             # Clustering and auxiliary objectives
│   └── unit.py             # Normalization, GRL schedule, and metrics
├── model.py                # ACGRL architecture
├── Training.py             # Two-stage training and evaluation pipeline
├── Run_model.py            # Main entry point
└── README.md
```

## Requirements

The implementation requires Python and the following packages:

```text
torch
numpy
scipy
scikit-learn
tqdm
```

Install the dependencies in an appropriate Python environment before running the experiments.

## Usage

Run an experiment from the project directory:

```bash
python Run_model.py --dataset NGs
```

The available dataset names are:

```text
NGs, BBCSport, Hdigit, Cora
```

For example:

```bash
python Run_model.py --dataset BBCSport --epochs 700 --pre_train 300 --seed 5
```

Display all available options with:

```bash
python Run_model.py --help
```

## Main Arguments

| Argument | Description | Default |
|---|---|---:|
| `--dataset` | Dataset used for training and evaluation | `NGs` |
| `--epochs` | Total number of training epochs | `700` |
| `--pre_train` | Number of Stage-I consistency-learning epochs | `300` |
| `--batch_size` | Batch size; uses the dataset configuration when omitted | Dataset-specific |
| `--train_rate` | Proportion of samples used for training | `0.8` |
| `--seed` | Random seed | `5` |
| `--lr` | Learning rate of the representation networks | `1e-4` |
| `--n_critic` | View-discriminator updates per iteration | `1` |
| `--epsilon` | Weight used by the complementarity objective | `0.8` |
| `--eta` | Entropy-regularization weight | `10.0` |
| `--beta` | GRL scheduling coefficient | `3.0` |
| `--lambda_clu` | Cluster-alignment loss weight | `0.5` |
| `--times` | Number of independent runs | `1` |

The Stage-I epoch count must be greater than zero and smaller than the total number of epochs.

## Training Procedure

1. **Adversarial consistency learning:** ACGRL trains a view discriminator and applies gradient reversal to remove view-identifiable information from the consistent representations.
2. **Complementarity learning:** The consistency branch is frozen, and view-specific encoders learn complementary residual information under consistency anchoring.
3. **Joint reconstruction:** Consistent and complementary representations are concatenated to reconstruct each input view.
4. **Clustering optimization:** View-wise cluster assignments are aligned through a cluster-level contrastive objective.
5. **Evaluation:** The trained model is evaluated once on the held-out set to avoid selecting an epoch using test labels.

## Evaluation Metrics

The program reports three standard clustering metrics:

- **ACC:** Clustering accuracy after Hungarian matching.
- **ARI:** Adjusted Rand Index.
- **NMI:** Normalized Mutual Information.

For multiple independent runs, the mean and standard deviation of ACC, ARI, and NMI are reported.

## Notes

- Dataset-specific input dimensions and default batch sizes are defined in `units/ConfigFunction.py`.
- The default protocol uses an 80/20 train-test split. When comparing with published results, ensure that the same evaluation protocol is used across all methods.
- The implementation focuses exclusively on multi-view clustering and does not contain classification or visualization pipelines.

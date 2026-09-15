import argparse


SUPPORTED_DATASETS = ("NGs", "BBCSport", "Hdigit", "Cora")


def parse_args():
    parser = argparse.ArgumentParser(description="Train ACGRL for multi-view clustering.")
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default="NGs")
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--pre_train", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--train_rate", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--n_critic", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=0.8)
    parser.add_argument("--eta", type=float, default=10.0)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--lambda_clu", type=float, default=0.5)
    parser.add_argument("--times", type=int, default=1)
    return parser.parse_args()
 

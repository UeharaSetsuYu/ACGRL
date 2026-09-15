import time

import numpy as np
import torch

from Training import main_train
from units.ConfigFunction import get_default_config
from units.ParserConfig import parse_args

def run_once(args, dataset_name, config, device):
    print("=" * 48)
    print(f"Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    print(f"Dataset: {dataset_name} ({config['view_num']} views)")
    print(f"Stage-I epochs: {args.pre_train}")
    print(f"Stage-II epochs: {args.epochs - args.pre_train}")
    print(f"Batch size: {args.batch_size}; seed: {args.seed}")
    return main_train(args, dataset_name, config, device)

def main():
    args = parse_args()
    config = get_default_config(args.dataset)
    if args.batch_size is None:
        args.batch_size = config['batch_size']
    if not 0 < args.train_rate < 1:
        raise ValueError("--train_rate must be between 0 and 1.")
    if not 0 < args.pre_train < args.epochs:
        raise ValueError("--pre_train must be greater than 0 and smaller than --epochs.")
    if args.lambda_clu < 0:
        raise ValueError("--lambda_clu must be non-negative.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {"acc": [], "ari": [], "nmi": []}
    initial_seed = args.seed

    for run_index in range(args.times):
        args.seed = initial_seed + 5 * run_index
        metrics = run_once(args, args.dataset, config, device)
        results["acc"].append(metrics["acc"])
        results["ari"].append(metrics["ari"])
        results["nmi"].append(metrics["nmi"])

    print("=" * 48)
    for name in ("acc", "ari", "nmi"):
        values = np.asarray(results[name], dtype=np.float64)
        print(f"{name.upper()}: mean={values.mean():.4f}, std={values.std():.4f}")


if __name__ == '__main__':
    start_time = time.time()
    main()
    print(f'Elapsed time: {time.time() - start_time:.2f}s')










from model import LitMNIST, torch_seed

import argparse

import pytorch_lightning as pl
import warnings
warnings.filterwarnings(action='ignore')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--order', type=int, default=None, help='seed order')
    parser.add_argument('--seed', type=int, default=223, help='set seed')
    parser.add_argument('--shuffle', type=bool, default=False, help='set dataloader shuffle')
    parser.add_argument('--epoch',type=int, default=30, help='set epoch')
    args = parser.parse_args()

    if args.order == 0:
        torch_seed(args.seed)

    # 1. Building model
    model = LitMNIST(hidden_size=64, learning_rate=0.0001, shuffle=args.shuffle)

    if args.order == 1:
        torch_seed(args.seed)

    # 2. creating Trainer
    trainer = pl.Trainer(
        gpus=1, 
        max_epochs=args.epoch, 
        progress_bar_refresh_rate=20
    )

    # 3. Training
    trainer.fit(model)


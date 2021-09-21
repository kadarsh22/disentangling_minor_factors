import wandb
import argparse
from src.config import get_config
from train import Trainer
from training_wrapper import run_training_wrapper
from evaluation_wrapper import run_evaluation_wrapper
from evaluation import Evaluator
from logger import PerfomanceLogger


def main():
    wandb.login(key='58f588faf95453b6f55ae88d33ede49f7805312f')
    parser = argparse.ArgumentParser(description='parser for project: disentangling minor factors of variation')
    parser.add_argument('--exp_name', help='name of the current run', required=True)
    parser.add_argument('--exp_desc', help='description/aim of this experiment', required=False, default='null')
    args = vars(parser.parse_args())
    config = get_config(args)
    PerfomanceLogger.configure_logger()
    perf_logger = PerfomanceLogger()
    if config.train:
        for seed in config.random_seeds:
            Trainer.set_seed(seed)
            run_training_wrapper(config, perf_logger)
    else:
        Evaluator.set_seed(config.random_seed)
        run_evaluation_wrapper(config, perf_logger)


if __name__ == "__main__":
    main()

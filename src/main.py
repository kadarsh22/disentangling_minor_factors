import wandb
from src.config import get_config
from train import Trainer
from training_wrapper import run_training_wrapper
from logger import PerfomanceLogger


def main():
    wandb.login(key='58f588faf95453b6f55ae88d33ede49f7805312f')
    config = get_config()
    Trainer.set_seed(config.random_seed)
    PerfomanceLogger.configure_logger()
    perf_logger = PerfomanceLogger()
    run_training_wrapper(config, perf_logger)


if __name__ == "__main__":
    main()

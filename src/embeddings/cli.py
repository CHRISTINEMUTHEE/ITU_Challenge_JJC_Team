import logging

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

from lightning.pytorch.cli import LightningCLI

import embeddings.dataloaders
import embeddings.models


def run_cli():
    cli = LightningCLI()


if __name__ == "__main__":
    run_cli()

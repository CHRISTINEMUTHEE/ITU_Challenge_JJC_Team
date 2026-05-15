import logging
import warnings

from lightning.pytorch.cli import LightningCLI

import embeddings.dataloaders
import embeddings.models

# Silence the Triton warning from torch.utils.flop_counter
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.flop_counter")
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)


def run_cli():
    cli = LightningCLI()


if __name__ == "__main__":
    run_cli()

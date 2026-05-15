from lightning.pytorch.cli import LightningCLI

import embeddings.dataloaders
import embeddings.models


def cli_main():
    cli = LightningCLI()


if __name__ == "__main__":
    cli_main()

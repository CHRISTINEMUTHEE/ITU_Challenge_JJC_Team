# Embeddings Model

## Setup

### Installation

This project uses `uv` for dependency management. Make sure you have it installed before running

On mac:
```bash
uv sync --dev
source venv/bin/activate
```

On windows:
```bash
uv sync --dev
.venv\Scripts\activate     
```

If you have any problems with installing pytorch, you can try some of the other options [here](https://docs.astral.sh/uv/guides/integration/pytorch/).

### Data Pre-processing

The data is in geotiff files right now, which are useless (for our purposes) and just make it slower to load. I've written some pre-processing steps in a jupyter notebook which you can follow. See [this notebook](notebooks/data_processing.ipynb)


## Training
I am writing this in pytorch lightning, so it should be fairly easy to use the cli to train things. I figured lightning would be better when we start mixing datasets and dataloader and models. See [here](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced.html) for the docs.

If you have a config file made up, you can use it to train a model like so.

```bash
embmodel fit --config .\configs\train_config.yaml
```
If you don't have a config file, you can follow the docs above to generate one.

This also lets you easily overwrite params from the command line, like in the example below, where we overwrite the max epochs.

```bash
embmodel fit --config .\configs\train_config.yaml --trainer.max_epochs 100
```

Training results are saved in the `logs/` directory. You can visualize them with tensorboard using 

```bash
tensorboard --logdir .\logs\[run name]
```

## Inference
I haven't done this yet. I am pretty sure you can just do this though.

```bash
embmodel predict --config .\configs\train_config.yaml
```

## Supported models. 

`SimpleConv`
[source](./src/embeddings/models/simpleconv.py)
This is the only model right now, it is a simple CNN with 3x(conv + batch norm + relu layers) and one linear conv layer as an output. It takes in CxBxHxW tensors and outputs 4xBxHxW tensors. It is trained using MSE with ADAM.

## Supported datasets.
Only the pixel level datasets are supported right now. (And I haven't done any preprocessing for the tessera embeddings). 
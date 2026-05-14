import pytest
import torch
from embeddings.models.simpleconv import CNN

@pytest.mark.parametrize("batch_size, n_channels, height, width, n_classes", [
    (1, 3, 64, 64, 10),
    (4, 1, 128, 128, 5),
    (2, 5, 32, 32, 2)
])
def test_cnn_output_shape(batch_size, n_channels, height, width, n_classes):
    model = CNN(
        n_channels=n_channels,
        n_classes=n_classes,
        kernel_size=3,
        padding="same",
        padding_mode="reflect"
    )
    
    x = torch.randn(batch_size, n_channels, height, width)
    y = model(x)
    
    # Output should be BxCxHxW
    assert y.shape == (batch_size, n_classes, height, width)

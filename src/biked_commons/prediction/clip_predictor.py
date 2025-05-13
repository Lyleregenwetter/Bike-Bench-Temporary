import torch
import torch.nn as nn
import dill

from biked_commons.resource_utils import models_and_scalers_path
from biked_commons.prediction.prediction_utils import TorchStandardScaler


def remove_wall_thickness(x, device):
    # indices_to_drop = [27, 28, 29, 30, 31, 32, 33]
    first_chunk = x[:, :27]
    second_chunk = x[:, 34:]
    x = torch.cat((first_chunk, second_chunk), dim=1)
    return x
    
class ResidualBlock(nn.Module):
    def __init__(self, input_size, layer_size, num_layers):
        super(ResidualBlock, self).__init__()
        self.layers = self._make_layers(input_size, layer_size, num_layers)

    def _make_layers(self, input_size, layer_size, num_layers):
        layers = [nn.Linear(input_size, layer_size), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(layer_size, layer_size))
            layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(layer_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.layers(x)
        total = out + residual
        return total


class ResidualNetwork(nn.Module):
    def __init__(self, input_size, output_size, layer_size, layers_per_block, num_blocks):
        super(ResidualNetwork, self).__init__()
        self.initial_layer = nn.Linear(input_size, layer_size)
        self.blocks = self._make_blocks(layer_size, layers_per_block, num_blocks)
        self.final_layer = nn.Linear(layer_size, output_size)
        

    def _make_blocks(self, layer_size, layers_per_block, num_blocks):
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(layer_size, layer_size, layers_per_block))
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.initial_layer(x)
        out = self.blocks(out)
        out = self.final_layer(out)
        return out

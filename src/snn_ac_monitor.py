import snntorch as snn
from torch import nn


# TODO: Docstrings
class SNNACMonitor:

    # TODO: Update later if needed
    SPIKING_LAYERS = (snn.Leaky,)

    def __init__(self, model):
        self.model = model

        self.layer_fanouts = self._calculate_fanouts()

    def _calculate_fanouts(self) -> dict[str, int]:
        fanouts = {}
        modules = list(self.model.named_modules())

        for i, (name, module) in enumerate(modules):
            if isinstance(module, self.SPIKING_LAYERS):
                for j in range(i + 1, len(modules)):
                    _, next_layer = modules[j]
                    fanout = self._calculate_fanout_for_layer(next_layer)
                    if fanout > 0:
                        fanouts[name] = fanout
                        break

        return fanouts

    @staticmethod
    def _calculate_fanout_for_layer(layer: nn.Module) -> int:
        print(layer)
        if isinstance(layer, nn.Linear):
            # A linear layer is FC, fanout is to all output neurons
            return layer.out_features
        elif isinstance(layer, nn.Conv2d):
            # For a conv layer, each pixel affects kernel_size^2 * out_channels neurons
            # This is an upper limit, as we don't take into account stride or padding (which would decrease fanout)
            # TODO: Do we need to take into account stride and padding?
            return layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
        else:
            # Otherwise return 0 as no weights to be updated in other layers
            return 0

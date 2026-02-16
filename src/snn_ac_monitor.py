import snntorch as snn
from torch import nn


# TODO: Docstrings
class SNNACMonitor:

    # TODO: Update later if needed
    SPIKING_LAYERS = (snn.Leaky,)

    def __init__(self, model: nn.Module):
        self.model = model
        self._layer_fanouts = self._calculate_fanouts(model)
        self._hooks = []
        self._total_acs = 0

    @staticmethod
    def _calculate_fanout_for_layer(layer: nn.Module) -> int:
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

    def _hook_callback(self, fanout: int):
        def hook(module, input, output) -> None:
            # Need just the spikes at index 0
            spikes = output[0].detach().sum().item()
            self._total_acs += spikes * fanout
        return hook

    def attach(self) -> None:
        for name, module in self.model.named_modules():
            if name in self._layer_fanouts:
                fanout = self._layer_fanouts[name]
                self._hooks.append(module.register_forward_hook(self._hook_callback(fanout)))

    def remove(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def get_total_acs(self) -> int:
        return self._total_acs

    @staticmethod
    def _calculate_fanouts(model: nn.Module) -> dict[str, int]:
        fanouts = {}
        modules = list(model.named_modules())

        for i, (name, module) in enumerate(modules):
            if isinstance(module, SNNACMonitor.SPIKING_LAYERS):
                for j in range(i + 1, len(modules)):
                    _, next_layer = modules[j]
                    fanout = SNNACMonitor._calculate_fanout_for_layer(next_layer)
                    if fanout > 0:
                        fanouts[name] = fanout
                        break

        return fanouts
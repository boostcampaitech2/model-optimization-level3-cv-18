from src.modules.base_generator import GeneratorAbstract

import torch
import torch.nn as nn

class Fire(nn.Module):
    """Codes from https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py#L18"""
    def __init__(self, 
                 inplanes: int, 
                 squeeze_planes: int, 
                 expand1x1_planes: int, 
                 expand3x3_planes: int
    ) -> None:
        super().__init__()
        self.inplanes = inplanes
        
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, 
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, 
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
                self.expand1x1_activation(self.expand1x1(x)), 
                self.expand3x3_activation(self.expand3x3(x))
        ], 1)
    
    
class FireGenerator(GeneratorAbstract):
    """Fire module generator for parsing module."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @property
    def base_module(self) -> nn.Module:
        """Returns module class from src.modules based on the class name."""
        return getattr(__import__("src.modules", fromlist=[""]), self.name)
    
    @property
    def out_channel(self) -> int:
        """Out channel of the module."""
        return int(self.args[1] + self.args[2]) # [1, Fire, [16, 64, 64]]

    def __call__(self, repeat: int = 1):
        """Returns nn.Module component"""
        args = [self.in_channel, *self.args]
        
        if repeat > 1:
            module = []
            for i in range(repeat):
                module.append(self.base_module(*args)) # Fire(*args)
                args[0] = self.out_channel
        else:
            module = self.base_module(*args)
        return self._get_module(module)
    

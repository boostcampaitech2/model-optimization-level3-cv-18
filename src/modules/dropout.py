import torch
import torch.nn as nn

from src.modules.base_generator import GeneratorAbstract

class Dropout(nn.Dropout):
    def __init__(self, p: float) -> None:
        super().__init__(p=p)
    

class DropoutGenerator(GeneratorAbstract):
    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        super().__init__(*args, **kwargs)
        self.p = 0.5
        if len(args) > 1:
            self.p = args[1]
        
    @property
    def base_module(self) -> nn.Module:
        """Returns module class from src.modules based on the class name."""
        return getattr(__import__("src.modules", fromlist=[""]), self.name)
    
    @property
    def out_channel(self) -> int:
        """Out channel of the module."""
        return self.in_channel

    def __call__(self, repeat: int = 1):
        """Returns nn.Module component"""
        return self._get_module(Dropout(self.p))    

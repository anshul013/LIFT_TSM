import torch
import torch.nn as nn
from models.TSMixer import Model as TSMixer
from models.LIFT import Model as LIFT

class Model(nn.Module):
    """
    LIFT-enhanced TSMixer model that uses leading indicators to refine predictions
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.backbone = TSMixer(configs)
        self.lift_wrapper = LIFT(
            backbone=self.backbone,
            configs=configs
        )
    
    def forward(self, x, *args, **kwargs):
        # LIFT wrapper handles both backbone prediction and refinement
        return self.lift_wrapper(x) 
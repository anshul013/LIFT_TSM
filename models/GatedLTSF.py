import torch
import torch.nn as nn
from models.Rev_in import RevIN

class Model(nn.Module):
    """
    Gated Time series model with Channel mixing
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.num_blocks = configs.num_blocks
        self.seq_len = configs.seq_len
        #Mixer block including gating
        self.channels = configs.enc_in
        self.pred_len = configs.pred_len
        self.rev_norm = RevIN(self.channels, affine=configs.affine)
        self.gating_layer = GatingLayer(self.seq_len)
        #Individual layer for each variate(if true) otherwise, one shared linear
        self.individual_linear_layers = configs.individual
        if(self.individual_linear_layers) :
            self.output_linear_layers = nn.ModuleList()
            for _ in range(self.channels):
                self.output_linear_layers.append(nn.Linear(configs.seq_len, configs.pred_len))
        else :
            self.output_linear_layers = nn.Linear(configs.seq_len, configs.pred_len)

    def forward(self, x) :
        x = self.rev_norm(x, 'norm')
        x = self.gating_layer(x)
        #Final linear layer applied on the transpoed mixers' output
        x = torch.swapaxes(x, 1, 2)
        #Preparing tensor output with the correct prediction length
        y = torch.zeros([x.size(0), x.size(1), self.pred_len],dtype=x.dtype).to(x.device)
        if self.individual_linear_layers :
            for c in range(self.channels): 
                y[:, c, :] = self.output_linear_layers[c](x[:, c, :].clone())
        else :
            y = self.output_linear_layers(x.clone())
        y = torch.swapaxes(y, 1, 2)
        y = self.rev_norm(y, 'denorm')
        return y

class GatingLayer(nn.Module) :
    """Gating Layer for timesteps importance sampling"""
    def __init__(self, seq_len) :
        super(GatingLayer, self).__init__()
        self.linear_gating = nn.Linear(seq_len, seq_len)
        self.sigmoid_activation = nn.Sigmoid()
    
    def forward(self, x) :
        x = torch.swapaxes(x, 1, 2)
        """Expecting single channel input"""
        gating_array = self.linear_gating(x)
        gating_array = self.sigmoid_activation(gating_array)
        y = torch.mul(x, gating_array)
        y = torch.swapaxes(y, 1, 2)
        return y
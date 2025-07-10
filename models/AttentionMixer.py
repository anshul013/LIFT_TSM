import torch
import torch.nn as nn
from models.Rev_in import RevIN

class Model(nn.Module):
    """
    Time Series Mixer model for modeling 
    channel & time steps interactions
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.num_blocks = configs.num_blocks
        self.mixer_block = MixerBlock(configs.enc_in, configs.seq_len, configs.dropout)
        self.channels = configs.enc_in
        self.pred_len = configs.pred_len
        self.rev_norm = RevIN(self.channels, affine=configs.affine)

        #Individual layer for each variate(if true) otherwise, one shared linear
        self.individual_linear_layers = configs.individual
        if(self.individual_linear_layers) :
            self.output_linear_layers = nn.ModuleList()
            for _ in range(self.channels):
                self.output_linear_layers.append(nn.Linear(configs.seq_len, configs.pred_len))
        else :
            self.output_linear_layers = nn.Linear(configs.seq_len, configs.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = self.rev_norm(x, 'norm')
        for _ in range(self.num_blocks):
            x = self.mixer_block(x)
        #Final linear layer applied on the transpoed mixers' output
        x = torch.swapaxes(x, 1, 2)
        # #Preparing tensor output with the correct prediction length
        y = torch.zeros([x.size(0), x.size(1), self.pred_len],dtype=x.dtype).to(x.device)
        if self.individual_linear_layers :
            for c in range(self.channels): 
                y[:, c, :] = self.output_linear_layers[c](x[:, c, :].clone())
        else :
            y = self.output_linear_layers(x.clone())
        y = torch.swapaxes(y, 1, 2)
        y = self.rev_norm(y, 'denorm')
        return y

    
class MlpBlockFeatures(nn.Module):
    """MLP for features"""
    def __init__(self, channels, timesteps, dropout_factor):
        super(MlpBlockFeatures, self).__init__()
        self.normalization_layer = nn.BatchNorm1d(channels)
        self.attention_layer = nn.MultiheadAttention(timesteps, 8, batch_first=True, dropout=dropout_factor)        

    def forward(self, x) :
        y = torch.swapaxes(x, 1, 2)
        # [batch_length, channel, time]
        y = self.normalization_layer(y)
        y, _ = self.attention_layer(y, y, y)
        y = torch.swapaxes(y, 1, 2)
        return x + y
    
class MlpBlockTimesteps(nn.Module):
    """MLP for features"""
    def __init__(self, channels, timesteps, dropout_factor):
        super(MlpBlockTimesteps, self).__init__()
        self.normalization_layer = nn.BatchNorm1d(timesteps)
        self.attention_layer = nn.MultiheadAttention(channels, 1, batch_first=True, dropout=dropout_factor)        

    def forward(self, x) :
        # [batch_length, time, Channel]
        y = self.normalization_layer(x)
        y, _ = self.attention_layer(y, y, y)
        return x + y
    
class MixerBlock(nn.Module):
    """Mixer block layer"""
    def __init__(self, channels, seq_len, dropout_factor) :
        super(MixerBlock, self).__init__()
        self.channels = channels
        self.seq_len = seq_len
        #Timesteps mixing block 
        self.timesteps_mixer = MlpBlockTimesteps(channels, seq_len, dropout_factor)
        #Features mixing block 
        self.channels_mixer = MlpBlockFeatures(channels, seq_len, dropout_factor)
    
    def forward(self, x) :
        y = self.timesteps_mixer(x)   
        y = self.channels_mixer(y)
        return y
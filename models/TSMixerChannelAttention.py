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
        self.mixer_block = MixerBlock(configs.enc_in, configs.seq_len, configs.dropout, configs.activation)
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
            x, attention_weights = self.mixer_block(x)
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
        return y, attention_weights

    
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
        y, attention_weights = self.attention_layer(y, y, y)
        y = torch.swapaxes(y, 1, 2)
        return x+y, attention_weights
    
class MlpBlockTimesteps(nn.Module):
    """MLP for timesteps with 1 layer"""
    def __init__(self, seq_len, dropout_factor, activation):
        super(MlpBlockTimesteps, self).__init__()
        self.normalization_layer = nn.BatchNorm1d(seq_len)
        self.linear_layer = nn.Linear(seq_len, seq_len)
        if activation=="gelu" :
            self.activation_layer = nn.GELU()
        elif activation=="relu" :
            self.activation_layer = nn.ReLU()
        else :
            self.activation_layer = None
        self.dropout_layer = nn.Dropout(dropout_factor)


    def forward(self, x) :
        y = self.normalization_layer(x)
        y = torch.swapaxes(y, 1, 2)
        y = self.linear_layer(y)
        y = self.activation_layer(y)
        y = self.dropout_layer(y)
        y = torch.swapaxes(y, 1, 2)
        return x + y
    
class MixerBlock(nn.Module):
    """Mixer block layer"""
    def __init__(self, channels, seq_len, dropout_factor, activation) :
        super(MixerBlock, self).__init__()
        self.channels = channels
        self.seq_len = seq_len
        #Timesteps mixing block 
        self.timesteps_mixer = MlpBlockTimesteps(seq_len, dropout_factor, activation)
        #Features mixing block 
        self.channels_mixer = MlpBlockFeatures(channels, seq_len, dropout_factor)
    
    def forward(self, x) :
        y = self.timesteps_mixer(x)   
        y, attention_weights = self.channels_mixer(y)
        return y, attention_weights
import torch
import torch.nn as nn
import math

from models.Rev_in import RevIN

class Model(nn.Module):
    """
    Patches-TSMixer based LTSF model mixing across channels and patches
    """
    def __init__(self, configs):
        super(Model, self).__init__()


        # Patching related parameters and layers
        self.patch_size = configs.patch_size
        self.seq_len = configs.seq_len
        self.channels = configs.enc_in
        self.stride = configs.stride
        self.num_patches = int((self.seq_len - self.patch_size)/self.stride + 1)
        self.rev_norm = RevIN(self.channels, configs.affine)
        self.hidden_size = configs.hidden_size


        # Mixing related parameters and layers
        self.mixer_block = MixerBlock(self.channels, self.hidden_size, self.num_patches, self.patch_size,
                                       configs.activation, configs.dropout,
                                       configs.exclude_inter_patch_mixing, configs.exclude_intra_patch_mixing,
                                       configs.exclude_channel_mixing)
        self.pred_len = configs.pred_len
        self.num_blocks = configs.num_blocks
        self.output_linear_layers = nn.Linear(self.num_patches*self.patch_size, self.pred_len)

    def forward(self, x):
        #  x : [batch size, channels, timesteps]
        x = self.rev_norm(x, "norm")
        x = torch.swapaxes(x, 1, 2)
        # Patching
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride) # reshaping to [batch size, channels, num_patches, patch_size]
        # Applying mixing step (keeps diminsions as is)
        for _ in range(self.num_blocks) :
            x = self.mixer_block(x)
        # Reshaping to [batch size, channels, timesteps]
        x = torch.reshape(x, 
                          (x.size(0),
                            self.channels,
                            self.num_patches*self.patch_size))
        # Preparing tensor output with the correct prediction length
        # Output tensor shape of [batch size, channel, pred_len]
        y = torch.zeros([x.size(0), self.channels, self.pred_len],dtype=x.dtype).to(x.device)
        y = self.output_linear_layers(x.clone())
        y = torch.swapaxes(y, 1, 2)
        y = self.rev_norm(y, "denorm")
        return y
        
class MlpBlockFeatures(nn.Module):
    """MLP for features"""
    def __init__(self, channels, hidden_size, activation, dropout_factor):
        super(MlpBlockFeatures, self).__init__()
        self.normalization_layer = nn.BatchNorm2d(channels)
        self.linear_layer = nn.Linear(channels, hidden_size)
        self.dropout_layer = nn.Dropout(dropout_factor)
        if activation=="gelu" :
            self.activation_layer = nn.GELU()
        elif activation=="relu" :
            self.activation_layer = nn.ReLU()
        else :
            self.activation_layer = None
        self.linear_layer2 = nn.Linear(hidden_size, channels)

    def forward(self, x) :
        # x : [batch_size, channels, num_patches, patch_size]
        y = self.normalization_layer(x)
        y = torch.swapaxes(y, 1, 3) # reshaping to [batch_size, patch_size, num_patches, channels]
        y = self.linear_layer(y)
        if self.activation_layer is not None :
            y = self.activation_layer(y)
        y = self.dropout_layer(y)
        y = self.linear_layer2(y)
        y = self.dropout_layer(y)
        y = torch.swapaxes(y, 1, 3) # reshaping back to [batch_size, channels, num_patches, patch_size]      
        return x + y
    
class MlpBlockPatches(nn.Module):
    """MLP for patches"""
    def __init__(self, num_patches, activation, dropout_factor):
        super(MlpBlockPatches, self).__init__()
        self.normalization_layer = nn.BatchNorm2d(num_patches)
        self.linear_layer = nn.Linear(num_patches, num_patches)
        self.dropout_layer = nn.Dropout(dropout_factor)
        if activation=="gelu" :
            self.activation_layer = nn.GELU()
        elif activation=="relu" :
            self.activation_layer = nn.ReLU()
        else :
            self.activation_layer = None


    def forward(self, x) :
        # [batch_size, channel, num_patches, patch_size]
        y = torch.swapaxes(x, 1, 2) # reshaping to [batch_size, num_patches, channels, patch_size]
        y = self.normalization_layer(y)
        y = torch.swapaxes(y, 1, 3) # reshaping to [batch_size, patch_size, channels, num_patches]

        y = self.linear_layer(y)
        if self.activation_layer is not None :
            y = self.activation_layer(y)
        y = self.dropout_layer(y)
        y = torch.swapaxes(y, 1, 3)
        y = torch.swapaxes(y, 1, 2) # reshaping back to same as input for residual connection : [batch_size, channel, num_patches, patch_size]
        return x + y    
    
class MlpBlockPatchSize(nn.Module):
    """MLP for num_patches"""
    def __init__(self, patch_size, activation, dropout_factor):
        super(MlpBlockPatchSize, self).__init__()
        self.normalization_layer = nn.BatchNorm2d(patch_size)
        self.linear_layer = nn.Linear(patch_size, patch_size)
        self.dropout_layer = nn.Dropout(dropout_factor)
        if activation=="gelu" :
            self.activation_layer = nn.GELU()
        elif activation=="relu" :
            self.activation_layer = nn.ReLU()
        else :
            self.activation_layer = None


    def forward(self, x) :
        # [batch_size, channels, num_patches, patch_size]
        y = torch.swapaxes(x, 1, 3) # reshaping to : [batch_size, patch_size, num_patches, channels]
        y = self.normalization_layer(y)
        y = torch.swapaxes(y, 1, 3) # reshaping back to : [batch_size, channels, num_patches, patch_size]
        y = self.linear_layer(y)
        if self.activation_layer is not None :
            y = self.activation_layer(y)
        y = self.dropout_layer(y)
        return x + y  

class MixerBlock(nn.Module):
    """Mixer block layer only mixing channels in this model"""
    def __init__(self, channels,hidden_size, num_patches, patch_size, activation, dropout_factor,
                 exclude_inter_patch_mixing, exclude_intra_patch_mixing, exclude_channel_mixing) :
        super(MixerBlock, self).__init__()
        self.exclude_inter_patch_mixing = exclude_inter_patch_mixing
        self.exclude_intra_patch_mixing = exclude_intra_patch_mixing
        self.exclude_channel_mixing = exclude_channel_mixing
        self.channels_mixer = MlpBlockFeatures(channels, hidden_size, activation, dropout_factor)
        self.patches_mixer = MlpBlockPatches(num_patches, activation, dropout_factor)
        self.patchSize_mixer = MlpBlockPatchSize(patch_size, activation, dropout_factor)

    def forward(self, x) :
        if(self.exclude_inter_patch_mixing and self.exclude_intra_patch_mixing and self.exclude_channel_mixing) :
            raise AttributeError("You can not set all mixing exclusion parameters to True")
        y = x
        if(not(self.exclude_channel_mixing)) :
            y = self.channels_mixer(y)
        if(not(self.exclude_inter_patch_mixing)) :
            y = self.patches_mixer(y)
        if(not(self.exclude_intra_patch_mixing)) :
            y = self.patchSize_mixer(y)
        return y


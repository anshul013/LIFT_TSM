import torch
import torch.nn as nn
from models.Rev_in import RevIN
import math

class Model(nn.Module):
    """
    Patches-Conv based LTSF model
    This model is based on the patching an input time series into equal sized contexts (patches)
    Then apply a mixing-based process on 3 different dimesnsions (Channels, Inter-patches, and Intra-patches)
    The mixing is done through convolutional operators
    """
    def __init__(self, configs):
        super(Model, self).__init__()

        # Patching related parameters and layers with a set of non-overlapping patches
        self.patch_size = configs.patch_size
        self.seq_len = configs.seq_len
        self.channels = configs.enc_in
        self.num_patches = math.ceil(self.seq_len / self.patch_size)
        self.remaining_timesteps = (self.num_patches * self.patch_size) - self.seq_len
        self.rev_norm = RevIN(self.channels, configs.affine)
        if self.remaining_timesteps == 0 :
            # No padding is needed
            self.replicationPadLayer = nn.ReplicationPad1d((0, 0))
            self.output_linear_layers = nn.Linear(configs.seq_len, configs.pred_len)

        else :
            padding_length = int(self.patch_size - self.remaining_timesteps)
            self.replicationPadLayer = nn.ReplicationPad1d((0, padding_length))
            # Adjusting the projection linear layer to accomodate the padded sequence
            self.output_linear_layers = nn.Linear(configs.seq_len+padding_length, configs.pred_len)

        # Mixing related parameters and layers
        self.mixer_block = MixerBlock(self.channels, self.num_patches, self.patch_size,
                                       configs.activation,
                                       configs.exclude_inter_patch_mixing, configs.exclude_intra_patch_mixing,
                                       configs.exclude_channel_mixing, configs.dropout)
        self.pred_len = configs.pred_len
        self.num_blocks = configs.num_blocks


    def forward(self, x):
        # Reshaping to [Batches, Channels, Timesteps]
        x = self.rev_norm(x, "norm")
        x = torch.swapaxes(x, 1, 2)
        # Repeating last values to make the number of timesteps divisible by the number of patches
        x = self.replicationPadLayer(x)
        num_patches = x.size(2) // self.patch_size
        # Reshaping the patching output to be suitable for the mixer block
        x = torch.reshape(x, 
                          (x.size(0),
                            x.size(1),
                            self.num_patches,
                            self.patch_size
                            ))
        # x dimension as input for mixer block : [batch_size, channels, num_pacthes, patch_size]
        for _ in range(self.num_blocks) :
            x = self.mixer_block(x) # Apply mixing block $num_blocks times
        # Reshaping to [batch_size, channels, padded_sequance]
        x = torch.reshape(x, 
                          (x.size(0),
                            self.channels,
                            num_patches*self.patch_size))
        # Preparing tensor output with the correct prediction length
        # Output tensor shape of [batch_size, channel, pred_len]
        y = torch.zeros([x.size(0), self.channels, self.pred_len],dtype=x.dtype).to(x.device)
        y = self.output_linear_layers(x.clone())
        y = torch.swapaxes(y, 1, 2)
        y = self.rev_norm(y, "denorm") # Finally denormalizing to finalize the RevIN operation
        return y
        
class ConvChannelMixer(nn.Module):
    """Conv block for channel mixing"""
    def __init__(self, channels, activation, dropout_factor):
        super(ConvChannelMixer, self).__init__()
        self.normalization_layer = nn.BatchNorm2d(channels)
        # Definition of convolution parameters
        kernel_size = 1
        channels_in = channels
        channels_out = channels
        self.conv_layer = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size)
        if activation=="gelu" :
            self.activation_layer = nn.GELU()
        elif activation=="relu" :
            self.activation_layer = nn.ReLU()
        else :
            self.activation_layer = None
        self.dropout_layer = nn.Dropout(dropout_factor)

    def forward(self, x) :
        # Swapping channel diminsion for applying normalization first
        x = self.normalization_layer(x)
        # Apply mixing through conv layer
        y = self.conv_layer(x)
        if self.activation_layer is not None :
            y = self.activation_layer(y)
        y = self.dropout_layer(y)
        return x + y
    
class ConvInterPatchMixer(nn.Module):
    """Conv block for inter-patch mixing"""
    def __init__(self, num_patches, activation, dropout_factor):
        super(ConvInterPatchMixer, self).__init__()
        self.normalization_layer = nn.BatchNorm2d(num_patches)
        # Definition of convolution parameters
        kernel_size = 1
        channels_in = num_patches
        channels_out = num_patches
        self.conv_layer = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size)        
        if activation=="gelu" :
            self.activation_layer = nn.GELU()
        elif activation=="relu" :
            self.activation_layer = nn.ReLU()
        else :
            self.activation_layer = None
        self.dropout_layer = nn.Dropout(dropout_factor)

    def forward(self, x) :
        # [batch_size, channels, num_patches, patch_size]
        x = torch.swapaxes(x, 1, 2)
        # [batch_size, num_patches, channels, patch_size]
        x = self.normalization_layer(x)
        y = self.conv_layer(x)
        if self.activation_layer is not None :
            y = self.activation_layer(y)
        y = self.dropout_layer(y)
        y = torch.swapaxes(y, 1, 2) # reverting the dimension back to match the input dimensions
        x = torch.swapaxes(x, 1, 2)
        return x + y    
    
class ConvIntraPatchMixer(nn.Module):
    """Conv block for Intra-patch mixing"""
    def __init__(self, patch_size, activation, dropout_factor):
        super(ConvIntraPatchMixer, self).__init__()
        self.normalization_layer = nn.BatchNorm2d(patch_size)
        # Definition of convolution parameters
        kernel_size = 1
        channels_in = patch_size
        channels_out = patch_size
        self.conv_layer = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size)        
        if activation=="gelu" :
            self.activation_layer = nn.GELU()
        elif activation=="relu" :
            self.activation_layer = nn.ReLU()
        else :
            self.activation_layer = None
        self.dropout_layer = nn.Dropout(dropout_factor)


    def forward(self, x) :
        # [batch_size, channels, num_patches, patch_size]
        x = torch.swapaxes(x, 1, 3)
        # [batch_size, patch_size, num_patches, channels]
        x = self.normalization_layer(x)
        y = self.conv_layer(x)
        if self.activation_layer is not None :
            y = self.activation_layer(y)
        y = self.dropout_layer(y)
        y = torch.swapaxes(y, 1, 3) # reverting the dimension back to match the input dimensions
        x = torch.swapaxes(x, 1, 3)
        return x + y  

class MixerBlock(nn.Module):
    """Mixer block layer mixing over a chosen subset of {channels, inter patches, intra patches}"""
    def __init__(self, channels, num_patches, patch_size, activation,
                 exclude_inter_patch_mixing, exclude_intra_patch_mixing, exclude_channel_mixing, dropout_factor) :
        super(MixerBlock, self).__init__()
        self.exclude_inter_patch_mixing = exclude_inter_patch_mixing
        self.exclude_intra_patch_mixing = exclude_intra_patch_mixing
        self.exclude_channel_mixing = exclude_channel_mixing
        self.channels_mixer    = ConvChannelMixer(channels, activation, dropout_factor)
        self.inter_patch_mixer = ConvInterPatchMixer(num_patches, activation, dropout_factor)
        self.inter_patch_mixer = ConvIntraPatchMixer(patch_size, activation, dropout_factor)

    def forward(self, x) :
        if(self.exclude_inter_patch_mixing and self.exclude_intra_patch_mixing and self.exclude_channel_mixing) :
            raise AttributeError("You can not set all mixing exclusion parameters to True")
        y = x
        # Adding the possibility to exclude mixer components from the model for ablation experimentation
        if(not(self.exclude_channel_mixing)) :
            y = self.channels_mixer(y)
        if(not(self.exclude_inter_patch_mixing)) :
            y = self.inter_patch_mixer(y)
        if(not(self.exclude_intra_patch_mixing)) :
            y = self.inter_patch_mixer(y)
        return y
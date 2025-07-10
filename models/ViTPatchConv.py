import torch
import torch.nn as nn
from models.Rev_in import RevIN

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
        self.num_channels = configs.enc_in
        self.embedding_dim = configs.embedding_dim
        self.activation = configs.activation
        self.hidden_dim = configs.hidden_size
        self.dropout = configs.dropout
        self.rev_norm = RevIN(self.num_channels, configs.affine)
        
        assert self.seq_len % self.patch_size == 0, "Sequence length should be divisble patch size"

        self.num_tokens = int(self.seq_len / self.patch_size)

        self.patch_embedding_layer= PatchEmbeddingLayer(self.patch_size, self.num_channels, self.embedding_dim, self.activation)

        # Mixing related parameters and layers
        self.mixer_block = MixerBlock(self.num_tokens, self.embedding_dim, self.hidden_dim, self.activation, self.dropout)
        self.pred_len = configs.pred_len
        self.num_blocks = configs.num_blocks

        self.channel_projection_layer = nn.Linear(self.embedding_dim, self.num_channels)
        self.pred_len_projection_layer = nn.Linear(self.num_tokens, self.pred_len)


    def forward(self, x):
        y = self.rev_norm(x, "norm")
        # Reshaping to [batch_size, channels, timesteps]
        y = torch.swapaxes(y, 1, 2)

        # Apply the per patch embedding layer
        y = self.patch_embedding_layer(y)

        # x dimension as input for mixer block : [batch_size, embedding dim, num of tokens]
        for _ in range(self.num_blocks) :
            y = self.mixer_block(y)     # output dimension of mixer block : [batch size, embedding dim, num of tokens]


        # Projection layer of num of tokens to the targeted forecasting sequence length
        y = self.pred_len_projection_layer(y) # output : [batch_size, embedding dim, pred_len]
        y = torch.swapaxes(y, 1, 2) # [batch_size,  pred_len, embedding dim]

        # Projection layer of embedding dim to the original feature space (num of channels)
        y = self.channel_projection_layer(y) # output : [batch_size, pred_len, channels]
        y = self.rev_norm(y, "denorm") # Finally denormalizing to finalize the RevIN operation for less distribution shift
        return y
        
class PatchEmbeddingLayer(nn.Module):
    """
    Conv block for embedding patches across channels into new embedding dimension
    Here the patching is done inherently through the conv operator
    Conv applied with (kernel size = patch_size, stride=patch_size, in_channels=num_channels, and out_channel=embedding_dim)
    """
    def __init__(self, patch_size, num_channels, embedding_dim, activation):
        super(PatchEmbeddingLayer, self).__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embedding_dim = embedding_dim
        self.activation = activation
        self.conv_layer = nn.Conv1d(self.num_channels, self.embedding_dim, self.patch_size, stride=self.patch_size)
        if activation=="gelu" :
            self.activation_layer = nn.GELU()
        elif activation=="relu" :
            self.activation_layer = nn.ReLU()
        else :
            self.activation_layer = None

    def forward(self, x) :
        y = self.conv_layer(x)
        if self.activation_layer is not None :
            y = self.activation_layer(y)
        # Expected output dimensions to be embedding_dim x num of tokens
        return y
    
class ConvChannelMixer(nn.Module):
    """Conv block for channel mixing"""
    def __init__(self, embedding_dim, hidden_dim, activation, dropout_factor):
        super(ConvChannelMixer, self).__init__()
        self.conv_layer1 = nn.Conv1d(embedding_dim, hidden_dim, 1)
        if activation=="gelu" :
            self.activation_layer = nn.GELU()
        elif activation=="relu" :
            self.activation_layer = nn.ReLU()
        else :
            self.activation_layer = None
        self.conv_layer2 = nn.Conv1d(hidden_dim, embedding_dim, 1)
        self.dropout_layer = nn.Dropout(dropout_factor)

    def forward(self, x) :
        # Apply mixing through conv layer
        y = self.conv_layer1(x)
        if self.activation_layer is not None :
            y = self.activation_layer(y)
        y = self.conv_layer2(y)
        y = self.dropout_layer(y)
        return y
    
class ConvTokenMixer(nn.Module):
    """Conv block for token mixing"""
    def __init__(self, num_tokens, hidden_dim, activation, dropout_factor):
        super(ConvTokenMixer, self).__init__()
        self.conv_layer1 = nn.Conv1d(num_tokens, hidden_dim, 1)
        if activation=="gelu" :
            self.activation_layer = nn.GELU()
        elif activation=="relu" :
            self.activation_layer = nn.ReLU()
        else :
            self.activation_layer = None
        self.conv_layer2 = nn.Conv1d(hidden_dim, num_tokens, 1)
        self.dropout_layer = nn.Dropout(dropout_factor)

    def forward(self, x) :
        # Apply mixing through conv layer
        y = self.conv_layer1(x)
        if self.activation_layer is not None :
            y = self.activation_layer(y)
        y = self.conv_layer2(y)
        y = self.dropout_layer(y)
        return y

class MixerBlock(nn.Module):
    """
    Mixer block layer mixing over a tokens then channels through conv-based mixer blocks
    Expects input with dims : [batch size, embedding dim, num of tokens(patches)]
    outputs the same dims as these of the input
    """
    def __init__(self, num_tokens, embedding_dim, hidden_dim, activation, dropout_factor) :
        super(MixerBlock, self).__init__()
        self.normalization_layer_tokens = nn.BatchNorm1d(num_tokens)        
        self.token_mixer = ConvTokenMixer(num_tokens, hidden_dim, activation, dropout_factor)
        self.normalization_layer_channels = nn.BatchNorm1d(embedding_dim)
        self.channels_mixer    = ConvChannelMixer(embedding_dim, hidden_dim, activation, dropout_factor)
        
    def forward(self, x) :
        # Dimensionality of input should be embedding dims x num of tokens
        y = torch.swapaxes(x, 1, 2)
        y = self.normalization_layer_tokens(y) # num of tokens x embedding dim
        x_T = torch.swapaxes(x, 1, 2)
        y =  x_T + self.token_mixer(y) # Apply the layer + a residual connection
        
        y = torch.swapaxes(y, 1, 2) # embedding dim x num of tokens
        y = self.normalization_layer_channels(y)
        y = self.channels_mixer(y)
        y = x + y
        # Dimensionality of output embedding dims x num of tokens
        return y
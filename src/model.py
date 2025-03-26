import torch
import torch.nn as nn
import math

from utils import Vocab

import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.3, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        
        x = x.transpose(0, 1)
        
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        
        x = x.transpose(0, 1)
        
        return x

class PositionWiseFFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        y = x
        y = self.fc1(x)
        y = torch.relu(x)
        y = self.fc2(x)
        return y
    
class AddAndNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, X, X_out):
        y = self.layer_norm(X + X_out)
        
        return y


class EncoderModule(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.3):
        super().__init__()
        
        self.attention_layer = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        
        self.add_and_norm_1 = AddAndNorm(d_model)
        
        self.position_wise = PositionWiseFFN(d_model)
        
        self.add_and_norm_2 = AddAndNorm(d_model)
    
    def forward(self, x, masks):        
        y = x
        
        y1, _ = self.attention_layer(y, y, y, need_weights=False, key_padding_mask=masks)
        
        y = self.add_and_norm_1(y, y1)
        
        y1 = self.position_wise(y)
        
        y = self.add_and_norm_2(y, y1)
        
        return y    
    
class Encoder(nn.Module):

    def __init__(self, encoder_len, d_model, num_heads):

        super().__init__()
        
        self.encoders = nn.ModuleList()
        self.n_modules = encoder_len
        self.d_model = d_model
        
        for _ in range(self.n_modules):
            self.encoders.append(EncoderModule(d_model, num_heads))
        
    def forward(self, x, masks):

        y = x

        for module in self.encoders:
            y = module(y, masks)
        
        return y


class DecoderModule(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.3):
        super().__init__()
    
        self.attention_layer_1 = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        
        self.add_and_norm_1 = AddAndNorm(d_model)
        
        self.attention_layer_2 = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        
        self.add_and_norm_2 = AddAndNorm(d_model)
        
        self.position_wise = PositionWiseFFN(d_model)
        
        self.add_and_norm_3 = AddAndNorm(d_model)
    
    def forward(self, x, encoder_output, attention_masks, key_padding_masks):
        
        y = x
        
        y1, _ = self.attention_layer_1(y, y, y, attn_mask=attention_masks, key_padding_mask=key_padding_masks, need_weights=False)
        
        y = self.add_and_norm_1(y, y1)
        
        y1, _ = self.attention_layer_2(y, encoder_output, encoder_output, need_weights=False)
        
        y = self.add_and_norm_2(y, y1)
        
        y1 = self.position_wise(y)
        
        y = self.add_and_norm_3(y, y1)
        
        return y
            

class Decoder(nn.Module):
    
    def __init__(self, decoder_len, d_model, num_heads):
        super().__init__()
        
        self.decoders = nn.ModuleList()
        self.n_modules = decoder_len
        self.d_model = d_model
        
        for _ in range(self.n_modules):
            self.decoders.append(DecoderModule(d_model, num_heads))
            
    def forward(self, x, encoder_output, attention_masks, key_padding_masks):
        
        y = x
        
        for module in self.decoders:
            y = module(y, encoder_output, attention_masks, key_padding_masks)
        
        return y
        

class Transformer(nn.Module):

    def __init__(self, d_model, d_input, d_output, n_encoder_stack, n_decoder_stack, num_heads):
        
        super().__init__()
        
        # init input and output embeddings
        self.input_embedding: nn.Embedding = nn.Embedding(d_input, d_model, padding_idx=0)
        
        self.output_embedding: nn.Embedding = nn.Embedding(d_output, d_model, padding_idx=0)

        self.d_model: int = d_model

        self.d_input: int = d_input
        
        self.d_output: int = d_output
        
        self.positional_encoding = PositionalEncoding(d_model)
        
        self.encoder = Encoder(encoder_len=n_encoder_stack, d_model=self.d_model, num_heads=num_heads)
        
        self.decoder = Decoder(decoder_len=n_decoder_stack, d_model=self.d_model, num_heads=num_heads)
        
        self.output_linear = nn.Linear(self.d_model, self.d_output)

        self._init_params()
        
    def forward(self, x, input_masks, outputs, attention_masks, key_padding_masks):
        
        # encoder
        y = x
        
        y = self.input_embedding(y)
        
        y = self.positional_encoding(y)
        
        encoder_output = self.encoder(y, input_masks)
        
        # decoder
        outputs = self.output_embedding(outputs)
        
        outputs = self.positional_encoding(outputs)
        
        y = self.decoder(outputs, encoder_output, attention_masks, key_padding_masks)
        
        y = self.output_linear(y)

        return y

    def get_parameters(self):
        
        p = []
        p.extend(list(self.encoder.parameters()))
        p.extend(list(self.decoder.parameters()))
        p.extend(list(self.output_linear.parameters()))
        p.extend(list(self.input_embedding.parameters()))
        p.extend(list(self.output_embedding.parameters()))
        p.extend(list(self.positional_encoding.parameters()))
        
        return p

    # TODO: implement
    def _init_params(self):
        ...
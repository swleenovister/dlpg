import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from common import constant
from base import InputEmbeddingLayer


class SASRec(nn.Module):
    def __init__(self, token_embedding, input_layer, n_items, input_dim, dim_feedforward, dropout_rate, device, activation='relu', n_head=1, n_blocks=2):
        '''
        SASRec dimension
        :param input_layer: base.InputEmbeddingLayer
        :param n_items: the number of all items.
        :param input_dim: dimension of item embedding(input embedding)
        :param dim_feedforward: hidden dimention of "Position-wise Feed Forward Network"
        :param dropout_rate:
        :param device:
        :param activation: default : "relu"  ("gelu" is another option for this)
        :param n_head: The number of heads (default is 1. followed the original paper setting)
        :param n_blocks: The number of heads (default is 2. followed the original paper setting)
        '''
        super(SASRec, self).__init__()
        self.token_embedding = token_embedding
        self._input_layer = input_layer
        self._transformer_encoder_layer = TransformerEncoderLayer(input_dim,
                                                                  n_head,
                                                                  dim_feedforward,
                                                                  dropout_rate,
                                                                  activation,
                                                                  device=device)
        self._transformer_encoder = TransformerEncoder(self._transformer_encoder_layer, n_blocks)
        self._output_layer = nn.Linear(input_dim, n_items)
        pass

    def _generate_mask(self, x):
        padding_mask =  x.transpose(0,1) == constant.TOKEN_IDX_PAD
        attention_mask = torch.tril(torch.ones(x.shape(0),x.shape(0)))
        padding_mask, attention_mask

    def forward(self, x, ):
        '''
        sequence length(SL), batch_size(B), embedding dimension(D)
        :param x:
        :return:
        '''
        # x : (SL, B)
        embedded_x = self._input_layer(x)

        # generate masking
        padding_mask, attention_mask = self._generate_padding_mask(x)

        # embedded_x : (SL, B, D)
        user_latent_embedding = self._transformer_encoder(embedded_x, attention_mask, padding_mask)

        # user_latent_embedding : (SL, B, D)
        self.token_embeddingg.weight
        return output


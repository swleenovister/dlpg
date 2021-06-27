'''
Transformer 구조를 따르는 모듈들에서 사용하는 기본 transformer 컴퍼넌트를 이 모듈 내에 정의함.
목록 (TBU)
'''
import torch
import torch.nn as nn

class InputEmbeddingLayer(nn.Module):
    def __init__(self, num_of_tokens : int,
                 emb_dim : int,
                 sequence_length : int,
                 static_embedding_tensors=None,
                 use_learnable_pos_embedding=False):
        '''
        TBU
        :param num_of_tokens:
        :param emb_dim:
        :param sequence_length:
        :param static_embedding_tensors:
        :param use_learnable_pos_embedding:
        '''
        super(InputEmbeddingLayer, self).__init__()

        # 필수 layer (token 혹은 item embedding layer 세팅
        self.sequence_length = sequence_length
        self.num_of_tokens = num_of_tokens
        self.emb_dim = emb_dim
        self.token_emb_layer = nn.Embedding(num_of_tokens, emb_dim)

        # # dropout layer 세팅 (FIXME : dropout은 활용적인 측면에서 밖에서 붙이는 게 나을 듯하여 없앰)
        # self.dropout_rate = dropout_rate
        # self.dropout_layer = nn.Dropout(dropout_rate)

        # 추가 add 되는 layer 세팅 (static positional embeddings)
        assert static_embedding_tensors is not None or use_learnable_pos_embedding, 'Positional Embedding MUST BE GIVEN.'

        # 차원이 전달된 파라미터와 동일한 지 확인.
        self.static_position_embedding_tensor = None
        if static_embedding_tensors is not None:
            for static_embedding_tensor in static_embedding_tensors:
                assert list(static_embedding_tensor.shape) == [sequence_length, emb_dim], 'The Provided Static Embedding Tensor has incorrect shape.'
            self.static_position_embedding_tensor = torch.sum(torch.stack(static_embedding_tensors), dim=0)

        # learnable embedding layers
        self.position_embedding_layer = None
        if use_learnable_pos_embedding is not None:
            self.position_embedding_layer = nn.Embedding(self.sequence_length, self.emb_dim)

    def forward(self, x, *args):
        '''
        sequence length(SL), batch size(B), embedding dimension(D)
        :param x: shape : (SL, B) - TOKEN INDEX TENSOR, this must be padded before injected to this function.
        :param args: ADDITIONAL EMBEDDING INDEX TENSOR, each must have
        :return:
        '''
        # TODO : batch_first 처리하기
        # TODO : 다양항 embedding 처리 가능하도록 함수 변경하기

        embedded = self.token_emb_layer(x) # (SL, B, D)

        if self.static_position_embedding_tensor is not None:
            embedded = (embedded.transpose(0,1) + self.static_position_embedding_tensor).transpose(0, 1)

        elif self.position_embedding_layer is not None:
            pos_emb = self.position_embedding_layer(torch.tensor(list(range(self.sequence_length))))
            embedded = (embedded.transpose(0,1) + pos_emb).transpose(0, 1)

        return embedded





if __name__ == '__main__':
    # bs = 3  # batch size for testing
    sl = 7  # sequence length for testing
    d = 5  # embedding dimention for testing
    vocab_size = 11  # vocabulary size for testing

    # InputEmbeddingLayer Test

    batch = torch.tensor([[0,0,0,3,1,4,4], [0,0,3,1,9,4,2]]).transpose(0,1).contiguous()
    emb_layer = InputEmbeddingLayer(vocab_size, d, sl, use_learnable_pos_embedding=True)
    out = emb_layer(batch)
    print(out)




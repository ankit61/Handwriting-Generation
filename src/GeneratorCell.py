from BaseModule import BaseModule
import torch
import torch.nn as nn
import constants
from Attention import Attention
from scipy.stats import ortho_group
import copy

class GeneratorCell(BaseModule):
    def __init__(self, invariant_size = constants.STYLE_VECTOR_SIZE, debug = True):
        super(GeneratorCell, self).__init__(debug)
        self.char_embedding = nn.Embedding(constants.CHARACTER_SET_SIZE, 
                                constants.CHARACTER_EMBEDDING_SIZE)
        self.invariant = nn.Embedding(constants.NUM_WRITERS, invariant_size)
        self.lstm_cell = nn.LSTMCell(invariant_size + \
                            constants.CHARACTER_EMBEDDING_SIZE, 
                            constants.LSTM_HIDDEN_SIZE)

        self.attn      = Attention(self.debug)
        self.init_embeddings()

    def init_embeddings(self):
        #set as many vectors to be orthogonal as possible
        with torch.no_grad():
            self.invariant.weight[:self.invariant.embedding_dim] = \
                torch.tensor(ortho_group.rvs(self.invariant.embedding_dim) \
                    [:self.invariant.num_embeddings])
        
            self.char_embedding.weight[:self.char_embedding.embedding_dim] = \
                torch.tensor(ortho_group.rvs(self.char_embedding.embedding_dim) \
                    [:self.char_embedding.num_embeddings])

        self.invariant.weight.requires_grad_()
        self.char_embedding.weight.requires_grad_()

    def forward(self, writer_id, letter_id_sequence, last_hidden, last_cell):
        invariants      = self.invariant(writer_id)
        attn_embedding  = self.attn(self.char_embedding(letter_id_sequence), last_hidden)
        lstm_input      = torch.cat([attn_embedding, invariants], dim=1)
        return self.lstm_cell(lstm_input, (last_hidden, last_cell))

    def introspect(self):
        pass
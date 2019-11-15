from BaseModule import BaseModule
import torch
import torch.nn as nn
import constants
from Attention import Attention
from scipy.stats import ortho_group
import copy

class GeneratorCell(BaseModule):
    def __init__(self, invariant_size = constants.STYLE_VECTOR_SIZE, lstm_depth = constants.LSTM_DEPTH, debug = True):
        super(GeneratorCell, self).__init__(debug)
        self.char_embedding = nn.Embedding(constants.CHARACTER_SET_SIZE, 
                                constants.CHARACTER_EMBEDDING_SIZE)
        self.invariant = nn.Embedding(constants.NUM_WRITERS, invariant_size)
        
        lstm_input_size = invariant_size + constants.CHARACTER_EMBEDDING_SIZE
        
        self.lstm_cells = [nn.LSTMCell(lstm_input_size, constants.LSTM_HIDDEN_SIZE)]
        for _ in range(lstm_depth - 1):
            self.lstm_cells.append(nn.LSTMCell(lstm_input_size + constants.LSTM_HIDDEN_SIZE, 
                                    constants.LSTM_HIDDEN_SIZE))

        self.lstm_cells = nn.ModuleList(self.lstm_cells)
        self.attn = Attention(self.debug)
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

    def forward(self, writer_id, letter_id_sequence, last_hidden_and_cell_states):
        assert len(last_hidden_and_cell_states) == len(self.lstm_cells), \
            f'last hidden and cell states ({len(last_hidden_and_cell_states)}) must be given for all lstms ({len(self.lstm_cells)})'
        invariants      = self.invariant(writer_id)
        attn_embedding  = self.attn(self.char_embedding(letter_id_sequence), last_hidden_and_cell_states[-1][0])

        hidden_and_cell_states = []
        lstm_input      = torch.cat([attn_embedding, invariants], dim=1)
        for i in range(len(self.lstm_cells)):
            hidden_and_cell_states.append(self.lstm_cells[i](lstm_input, last_hidden_and_cell_states[i]))
            lstm_input = torch.cat([attn_embedding, invariants, hidden_and_cell_states[-1][0]], dim=1)

        return hidden_and_cell_states

    def introspect(self):
        pass
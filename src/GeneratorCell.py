from BaseModule import BaseModule
import torch
import torch.nn as nn
import constants
from Attention import Attention
from scipy.stats import ortho_group
import copy

class GeneratorCell(BaseModule):
    def __init__(self, invariant_size = constants.STYLE_VECTOR_SIZE, rnn_type = constants.RNN_TYPE, 
        debug = True):
        super(GeneratorCell, self).__init__(debug)
        assert rnn_type == 'LSTM' or rnn_type == 'GRU', \
            'rnn_type should be \'LSTM\' or \'GRU\''
        assert constants.RNN_DEPTH > 0
        
        self.char_embedding = nn.Embedding(constants.CHARACTER_SET_SIZE, 
                                constants.CHARACTER_EMBEDDING_SIZE)
        self.invariant = nn.Embedding(constants.NUM_WRITERS, invariant_size)
        self.rnn_type = rnn_type
        rnn_cell_type = nn.GRUCell if rnn_type == 'GRU' else nn.LSTMCell
        rnn_input_size = invariant_size + constants.CHARACTER_EMBEDDING_SIZE + constants.RNN_OUT_SIZE
        
        self.rnn_cells = nn.ModuleList([rnn_cell_type(rnn_input_size, constants.RNN_HIDDEN_SIZE)])
        for _ in range(constants.RNN_DEPTH - 1):
            self.rnn_cells.append(rnn_cell_type(rnn_input_size + constants.RNN_HIDDEN_SIZE, 
                                    constants.RNN_HIDDEN_SIZE))

        self.attn = Attention(self.debug)
        self.fc   = nn.Linear(constants.RNN_HIDDEN_SIZE * constants.RNN_DEPTH + constants.RNN_OUT_SIZE, constants.RNN_OUT_SIZE)
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

    def forward(self, writer_id, letter_id_sequence, last_hidden_and_cell_states, last_out):
        assert len(last_hidden_and_cell_states) == len(self.rnn_cells), \
            f'last hidden and cell states ({len(last_hidden_and_cell_states)}) must be given for all rnn cells ({len(self.rnn_cells)})'

        if(torch.cuda.is_available()):
            writer_id = writer_id.cuda()
            letter_id_sequence = letter_id_sequence.cuda()
            last_out = last_out.cuda()

        invariants      = self.invariant(writer_id)
        attn_embedding  = self.attn(self.char_embedding(letter_id_sequence), 
            [last_hidden_and_cell_states[i][0] for i in range(len(last_hidden_and_cell_states))])

        hidden_and_cell_states = []
        rnn_input      = torch.cat([attn_embedding, invariants, last_out], dim=1)
        for i in range(len(self.rnn_cells)):
            if self.rnn_type == 'GRU':
                rnn_out = self.rnn_cells[i](rnn_input, last_hidden_and_cell_states[i][0])
                hidden_and_cell_states.append((rnn_out, None))
            else: # LSTM
                rnn_out = self.rnn_cells[i](rnn_input, last_hidden_and_cell_states[i])
                hidden_and_cell_states.append(rnn_out)

            rnn_input = torch.cat([attn_embedding, invariants, last_out, hidden_and_cell_states[-1][0]], dim=1)

        final_in  = torch.cat(
            [last_out] + [hidden_and_cell_states[i][0] for i in range(constants.RNN_DEPTH)], dim=1)
        final_out = self.fc(final_in)

        return final_out, hidden_and_cell_states

    def introspect(self):
        pass

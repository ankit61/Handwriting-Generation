from BaseModule import BaseModule
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import ortho_group
import copy
import constants
from Attention import Attention, WindowAttention

class GeneratorCell(BaseModule):
    def __init__(self, invariant_size = constants.STYLE_VECTOR_SIZE, rnn_type = constants.RNN_TYPE, 
        debug = True):
        super(GeneratorCell, self).__init__(debug)
        assert rnn_type == 'LSTM' or rnn_type == 'GRU', \
            'rnn_type should be \'LSTM\' or \'GRU\''
        assert constants.RNN_DEPTH > 0
        

        self.char_embedding = nn.Embedding(constants.CHARACTER_SET_SIZE, 
                                constants.CHARACTER_SET_SIZE) # One-hot encoding
        self.invariant = nn.Embedding(constants.NUM_WRITERS, invariant_size)
        self.rnn_type = rnn_type
        rnn_cell_type = nn.GRUCell if rnn_type == 'GRU' else nn.LSTMCell
        rnn_input_size = invariant_size + constants.RNN_OUT_SIZE
        
        self.rnn_cells = nn.ModuleList([rnn_cell_type(rnn_input_size, constants.RNN_HIDDEN_SIZE)])
        for _ in range(constants.RNN_DEPTH - 1):
            self.rnn_cells.append(rnn_cell_type(rnn_input_size + constants.RNN_HIDDEN_SIZE, 
                                constants.RNN_HIDDEN_SIZE))

        self.attn = WindowAttention(hidden_size=constants.RNN_DEPTH * constants.RNN_HIDDEN_SIZE ,debug=self.debug)
        self.fc   = nn.Linear(constants.RNN_HIDDEN_SIZE * constants.RNN_DEPTH + constants.RNN_OUT_SIZE, constants.RNN_OUT_SIZE)
        self.init_embeddings()

    def init_embeddings(self):
        #set as many vectors to be orthogonal as possible
        with torch.no_grad():
            self.invariant.weight[:self.invariant.embedding_dim] = \
                torch.tensor(ortho_group.rvs(self.invariant.embedding_dim) \
                    [:self.invariant.num_embeddings])
        
            # Use identity matrix for one-hot encoding
            self.char_embedding.weight[:self.char_embedding.embedding_dim] = \
                torch.tensor(np.identity(self.char_embedding.embedding_dim))

        self.invariant.weight.requires_grad_()
        self.char_embedding.weight.requires_grad_(False)

    def forward(self, writer_id, letter_id_sequence, orig_text_lens, last_hidden_and_cell_states, last_out, last_kappa):
        assert len(last_hidden_and_cell_states) == len(self.rnn_cells), \
            f'last hidden and cell states ({len(last_hidden_and_cell_states)}) must be given for all rnn cells ({len(self.rnn_cells)})'

        if(torch.cuda.is_available()):
            writer_id = writer_id.cuda()
            letter_id_sequence = letter_id_sequence.cuda()
            last_out = last_out.cuda()

        last_out[:, 2].sigmoid_()
        invariants      = self.invariant(writer_id)

        # Concatenate hidden states into 1D vector
        #attn_hidden_states = [last_hidden_and_cell_states[i][0] for i in range(len(last_hidden_and_cell_states))]
        #attn_hidden_states = torch.cat([attn_hidden_states[i] for i in range(len(attn_hidden_states))], dim=1)
        #attn_embedding, cur_kappa  = self.attn(self.char_embedding(letter_id_sequence), attn_hidden_states, last_kappa, orig_text_lens)
        cur_kappa = None
        
        # Only use the first 'batch size' attentions as sequences in a batch are different length
        #attn_embedding = attn_embedding[:last_out.shape[0]]

        hidden_and_cell_states = []
        rnn_input      = torch.cat([invariants, last_out], dim=1)
        for i in range(len(self.rnn_cells)):
            if self.rnn_type == 'GRU':
                rnn_out = self.rnn_cells[i](rnn_input, last_hidden_and_cell_states[i][0])
                hidden_and_cell_states.append((rnn_out, None))
            else: # LSTM
                rnn_out = self.rnn_cells[i](rnn_input, last_hidden_and_cell_states[i])
                hidden_and_cell_states.append(rnn_out)
            rnn_input = torch.cat([invariants, last_out, hidden_and_cell_states[-1][0]], dim=1)

        final_in  = torch.cat(
            [last_out] + [hidden_and_cell_states[i][0] for i in range(constants.RNN_DEPTH)], dim=1)
        final_out = self.fc(final_in)

        return final_out, hidden_and_cell_states, cur_kappa

    def introspect(self):
        pass

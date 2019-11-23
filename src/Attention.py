from BaseModule import BaseModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import constants

class Attention(BaseModule):
    def __init__(self,  debug = True):
        super(Attention, self).__init__(debug)
        self.attn = nn.Linear(constants.CHARACTER_EMBEDDING_SIZE * 
            constants.MAX_LINE_TEXT_LENGTH + constants.RNN_HIDDEN_SIZE * constants.RNN_DEPTH, 
            constants.MAX_LINE_TEXT_LENGTH)
        self.attn_weights = None
        # self.abs_coords = None

    def forward(self, letter_embedding_sequence, orig_text_lens, last_hidden_states, last_out):
        #letter_embedding_sequence.shape -> batch_size x constants.MAX_LINE_POINTS x 
        #                                   constants.CHARACTER_EMBEDDING_SIZE

        # if self.abs_coords is None:
        #     self.abs_coords = last_out
        # else:
        #     self.abs_coords[:, :2] += last_out[:, :2]
        #     self.abs_coords[:, 2:] += last_out[:, 2:]

        flattened_sequence = letter_embedding_sequence.view(
                                letter_embedding_sequence.shape[0], -1)

        last_hidden_concat = torch.cat([last_hidden_states[i] for i in range(len(last_hidden_states))], dim=1)
        # attn_input = torch.cat([flattened_sequence, last_hidden_concat, self.abs_coords], dim=1)
        attn_input = torch.cat([flattened_sequence, last_hidden_concat], dim=1)
        attn_out   = F.relu(self.attn(attn_input))

        #make it impossible to pay attention to padding char
        for i in range(orig_text_lens.shape[0]): 
            attn_out[:, orig_text_lens[i]:] = 0

        self.attn_weights = F.softmax(attn_out, dim=1).unsqueeze(1)

        #attn_weights.shape -> batch_size x 1 x constants.MAX_LINE_POINTS

        return torch.bmm(self.attn_weights, letter_embedding_sequence).squeeze(1)

    def introspect(self):
        return

class WindowAttention(BaseModule):
    def __init__(self, hidden_size=constants.RNN_HIDDEN_SIZE, num_gaussian_func=constants.ATTENTION_NUM_GAUSSIAN_FUNC, 
            max_text_len=constants.MAX_LINE_TEXT_LENGTH, debug=True):
        super(WindowAttention, self).__init__(debug)

        self.attn = nn.Linear(hidden_size, 3*num_gaussian_func) # 3 for predicting alpha, beta & kappa
        self.max_text_len = max_text_len
        self.hidden_size = hidden_size
        self.num_gaussian_func = num_gaussian_func
        self.last_letter_weights = None

    def get_attn_weights(self):
        return self.last_letter_weights

    def forward(self, letter_embedding_sequence, last_hidden_states, last_kappa, text_len):
        assert last_hidden_states.shape[-1] == self.hidden_size, \
            f'Hidden size for attention ({self.hidden_size}) must match size of last_hidden_states ({last_hidden_states.shape[-1]})'
        assert last_kappa.shape[-1] == self.num_gaussian_func, \
            f'Number of gaussian functions for attention ({self.num_gaussian_func}) must match shape of last_kappa ({last_kappa.shape[-1]})'

        attn_params = self.attn(last_hidden_states).exp()

        # Zero out embeddings for padding characters
        for i in range(text_len.shape[0]):
            letter_embedding_sequence[:, text_len[i]:] = 0

        
        alpha, beta, cur_kappa = attn_params.chunk(3, dim=-1)
        kappa = last_kappa + cur_kappa

        character_indices = torch.Tensor(list(range(self.max_text_len)))
        
        exponent = - beta.unsqueeze(2) * (kappa.unsqueeze(2).repeat(1, 1, self.max_text_len) - character_indices)**2
        letter_weights = (alpha.unsqueeze(2) * torch.exp(exponent)).sum(dim=1)
        attn_out = (letter_weights.unsqueeze(2) * letter_embedding_sequence).sum(dim=1)

        self.last_letter_weights = letter_weights
        return attn_out, kappa
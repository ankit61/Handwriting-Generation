from BaseModule import BaseModule
import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def forward(self, letter_embedding_sequence, last_hidden_states, prev_kappa, text_len):
        attn_params = self.attn(last_hidden_states)
        alpha, beta, cur_kappa = attn_params.chunk(3, dim=-1)

        print(letter_embedding_sequence.shape)
        exit(-1)
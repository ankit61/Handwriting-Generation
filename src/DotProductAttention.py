from BaseModule import BaseModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import constants

class DotProductAttention(BaseModule):
    def __init__(self, query_size=constants.RNN_HIDDEN_SIZE, debug = True):
        super(DotProductAttention, self).__init__(debug)
        self.query_size = query_size
        self.query_weights = None

        assert self.query_weights != None or self.query_size == constants.CHARACTER_EMBEDDING_SIZE, \
          f'Query size {self.query_size} must match character embedding size {constants.CHARACTER_EMBEDDING_SIZE} when query_weights are not used'

    def forward(self, letter_embedding_sequence, query, orig_text_lens):
        #letter_embedding_sequence.shape -> batch_size x constants.MAX_LINE_POINTS x 
        #                                   constants.CHARACTER_EMBEDDING_SIZE

        # if self.abs_coords is None:
        #     self.abs_coords = last_out
        # else:
        #     self.abs_coords[:, :2] += last_out[:, :2]
        #     self.abs_coords[:, 2:] += last_out[:, 2:]

        for i in range(orig_text_lens.shape[0]): 
            letter_embedding_sequence[:, orig_text_lens[i]:] = 0

        query = torch.unsqueeze(query, dim=1)
        attn_keys = torch.transpose(letter_embedding_sequence, dim0=1, dim1=2)
        logit_weights = torch.bmm(query, attn_keys)
        value_weights = F.softmax(logit_weights, dim=2)
        attn_out = torch.bmm(value_weights, letter_embedding_sequence).squeeze(dim=1)

        return attn_out

    def introspect(self):
        return
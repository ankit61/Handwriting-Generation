from BaseModule import BaseModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import constants

class Attention(BaseModule):
    def __init__(self,  debug = True):
        super(Attention, self).__init__(debug)
        self.attn = nn.Linear(constants.CHARACTER_EMBEDDING_SIZE * 
            constants.MAX_LINE_TEXT_LENGTH + constants.LSTM_HIDDEN_SIZE, 
            constants.MAX_LINE_POINTS)

    def forward(self, letter_embedding_sequence, last_hidden):
        flattened_sequence = letter_embedding_sequence.view(
                                letter_embedding_sequence.shape[0], -1)
        if last_hidden is None:
            last_hidden = torch.zeros(letter_embedding_sequence.shape[0], 
                            constants.LSTM_HIDDEN_SIZE)
        attn_input = torch.cat([flattened_sequence, last_hidden], dim=1)
        attn_weights = F.softmax(self.attn(flattened_sequence), dim=1).unsqueeze(1)
        #attn_weights.shape -> batch_size x 1 x constants.MAX_LINE_POINTS
        #letter_embedding_sequence.shape -> batch_size x constants.MAX_LINE_POINTS x 
        #                                   constants.CHARACTER_EMBEDDING_SIZE
        return torch.bmm(attn_weights, letter_embedding_sequence).squeeze(1)

    def introspect(self):
        return
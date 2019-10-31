import BaseModule
import torch
import torch.nn as nn
import constants

HIDDEN_SIZE = 3

class GeneratorCell(BaseModule):
    def __init__(self, character_set_size = constants.CHARACTER_SET_SIZE, invariant_size = constants.STYLE_VECTOR_SIZE, debug = True):
        super(GeneratorCell, self).__init__(debug)
        self.invariant = torch.randn(invariant_size, requires_grad=True)
        self.lstm_cell = nn.LSTMCell(character_set_size + invariant_size, 
                            HIDDEN_SIZE)
    
    def forward(self, encoded_letter, hx=None):
        x = torch.cat(encoded_letter, self.invariant)
        return lstm_cell(x, hx)

    def introspect(self):
        pass
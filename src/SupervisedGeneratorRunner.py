from GeneratorCell import GeneratorCell
from BaseRunner import BaseRunner
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import constants

LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

class SupervisedGeneratorRunner(BaseRunner):
    def __init__(self, debug = True):
        model = GeneratorCell()
        optimizer = optim.SGD(lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        super(SupervisedGeneratorRunner, self).__init__(models=[model],
            loss_fn=SupervisedGeneratorRunner.generator_loss, 
            optimizers=[optimizer], best_metric_name='loss', 
            should_minimize_best_metric=True)

    def run_batch_and_get_metrics(self, batch, is_train_mode):
        #batch['datapoints'].shape -> batch_size x max_seq_len x features_of_element
        packed_datapoints = rnn_utils.pack_padded_sequence(batch['datapoints'],
            batch['orig_datapoints_len'], batch_first=True)

        batch_start = 0
        last_hidden = torch.zeros(batch.shape[0], constants.LSTM_HIDDEN_SIZE)
        last_cell   = torch.zeros(batch.shape[0], constants.LSTM_HIDDEN_SIZE)
        
        loss = torch.tensor([0.0], requires_grad=is_train_mode)
        self.optimizers[0].zero_grad()
        
        for i, cur_batch_size in enumerate(packed_datapoints.batch_sizes):
            #do forward pass
            letter_id_sequences = batch['line_text_integers'][:cur_batch_size, :]
            writer_ids = batch['writer_id'][:cur_batch_size]

            last_hidden = last_hidden[:cur_batch_size, :]
            last_cell   = last_cell[:cur_batch_size, :]

            last_hidden, last_cell = self.nets[0](letter_id_sequences, writer_ids,
                                                    (last_hidden, last_cell))

            #compute loss
            gt = packed_datapoints.data[batch_start:batch_start + cur_batch_size, :]
            generated = last_hidden[:cur_batch_size, :3]
            loss += self.loss_fn(generated, gt)

            if is_train_mode:
                #calculate gradients but don't update
                loss.backward()

            batch_start += cur_batch_size

        if(is_train_mode):
            #update weights
            self.optimizers[0].step()

        return [('loss', loss.mean().item())]

    def train_batch_and_get_metrics(self, batch):
        self.run_batch_and_get_metrics(batch, is_train_mode=True)

    def test_batch_and_get_metrics(self, batch):
        self.run_batch_and_get_metrics(batch, is_train_mode=False)
    
    @staticmethod
    def generator_loss(generated, gt):
        xys  = generated.narrow(1, 0, 2)
        ps   = generated.narrow(1, 2, 1)

        mse_loss = nn.MSELoss()
        bce_loss = nn.BCEWithLogitsLoss()
        return mse_loss(xys, gt.narrow(2, 0, 2)) + bce_loss(ps, gt.narrow(2, 2, 1))
from GeneratorCell import GeneratorCell
from BaseRunner import BaseRunner
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import constants
import torch
from tqdm import tqdm

LR = 0.0001
MOMENTUM = 0.9
WEIGHT_DECAY = 0
GRADIENT_CLIP_NORM = 5

class SupervisedGeneratorRunner(BaseRunner):
    def __init__(self, debug = True):
        model = GeneratorCell()
        optimizer = optim.Adam(model.parameters())
        self.global_step = 0
        super(SupervisedGeneratorRunner, self).__init__(models=[model],
            loss_fn=SupervisedGeneratorRunner.generator_loss, 
            optimizers=[optimizer], best_metric_name='loss', 
            should_minimize_best_metric=True)
        self.set_gpu_keys(['datapoints', 'writer_id', 'line_text_integers'])

    def run_batch_and_get_metrics(self, batch, is_train_mode):
        #batch['datapoints'].shape -> batch_size x max_seq_len x features_of_element
        packed_datapoints = rnn_utils.pack_padded_sequence(batch['datapoints'],
            batch['orig_datapoints_len'].cpu(), batch_first=True)

        batch_start = 0
        last_hidden = torch.zeros(batch['datapoints'].shape[0], constants.LSTM_HIDDEN_SIZE)
        last_cell   = torch.zeros(batch['datapoints'].shape[0], constants.LSTM_HIDDEN_SIZE)
        if torch.cuda.is_available():
            last_hidden, last_cell = last_hidden.cuda(), last_cell.cuda()

        loss = 0.0
        self.optimizers[0].zero_grad()

        for i, cur_batch_size in tqdm(enumerate(packed_datapoints.batch_sizes)):
            #do forward pass
            letter_id_sequences = batch['line_text_integers'][:cur_batch_size, :]
            writer_ids = batch['writer_id'][:cur_batch_size]

            last_hidden = last_hidden[:cur_batch_size, :]
            last_cell   = last_cell[:cur_batch_size, :]

            last_hidden, last_cell = self.nets[0](writer_ids, letter_id_sequences,
                                                    last_hidden, last_cell)

            #compute loss
            gt = packed_datapoints.data[batch_start:batch_start + cur_batch_size, :]
            generated = last_hidden[:cur_batch_size, :3]
            loss += self.loss_fn(generated, gt)
            if is_train_mode:
                #calculate gradients but don't update
                loss.backward(retain_graph=(i < len(packed_datapoints.batch_sizes) - 1))
                torch.nn.utils.clip_grad_norm_(self.nets[0].parameters(), GRADIENT_CLIP_NORM)
                self.output_gradient_distributions(self.global_step)
                self.global_step += 1

            batch_start += cur_batch_size

        if(is_train_mode):
            #update weights
            self.optimizers[0].step()

        return [('loss', loss.div_(len(packed_datapoints.batch_sizes)).item())]

    def train_batch_and_get_metrics(self, batch):
        return self.run_batch_and_get_metrics(batch, is_train_mode=True)

    def test_batch_and_get_metrics(self, batch):
        return self.run_batch_and_get_metrics(batch, is_train_mode=False)

    @staticmethod
    def generator_loss(generated, gt):
        xys  = generated.narrow(1, 0, 2)
        ps   = generated.narrow(1, 2, 1)

        mse_loss = nn.MSELoss().cuda() if torch.cuda.is_available() else nn.MSELoss()
        bce_loss = nn.BCEWithLogitsLoss().cuda() if torch.cuda.is_available() else nn.BCEWithLogitsLoss()
        return mse_loss(xys, gt.narrow(1, 0, 2)) + bce_loss(ps, gt.narrow(1, 2, 1))

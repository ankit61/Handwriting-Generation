from GeneratorCell import GeneratorCell
from BaseRunner import BaseRunner
from utils import delta_points_to_image
import constants
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import constants
import torch
from tqdm import tqdm

LR = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0
GRADIENT_CLIP_NORM = 5
UPDATE_BATCHES_PERIOD = 20

class SupervisedGeneratorRunner(BaseRunner):
    def __init__(self, debug = True):
        model = GeneratorCell()
        optimizer = optim.Adam(model.parameters(), lr=LR)
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
        last_hidden_and_cell_states = []

        #initialize last_hidden_and_cell_states
        for _ in range(constants.LSTM_DEPTH):
            last_hidden = torch.zeros(batch['datapoints'].shape[0], constants.LSTM_HIDDEN_SIZE)
            last_cell   = torch.zeros(batch['datapoints'].shape[0], constants.LSTM_HIDDEN_SIZE)

            if torch.cuda.is_available():
                last_hidden, last_cell = last_hidden.cuda(), last_cell.cuda()

            last_hidden_and_cell_states.append((last_hidden, last_cell))

        loss = 0.0
        self.optimizers[0].zero_grad()

        for i, cur_batch_size in tqdm(enumerate(packed_datapoints.batch_sizes)):
            #do forward pass
            letter_id_sequences = batch['line_text_integers'][:cur_batch_size, :]
            writer_ids = batch['writer_id'][:cur_batch_size]

            for j in range(constants.LSTM_DEPTH):
                last_hidden_and_cell_states[j] = (last_hidden_and_cell_states[j][0][:cur_batch_size, :], 
                                                  last_hidden_and_cell_states[j][1][:cur_batch_size, :])

            last_hidden_and_cell_states = self.nets[0](writer_ids, letter_id_sequences,
                                                    last_hidden_and_cell_states)

            #compute loss
            gt = packed_datapoints.data[batch_start:batch_start + cur_batch_size, :]
            generated = last_hidden_and_cell_states[-1][0][:, :3]
            loss += self.loss_fn(generated, gt, self.writer, self.global_step)
            if is_train_mode:
                #calculate gradients but don't update
                loss.backward(retain_graph=(i < len(packed_datapoints.batch_sizes) - 1))
                torch.nn.utils.clip_grad_norm_(self.nets[0].parameters(), GRADIENT_CLIP_NORM)
                self.global_step += 1
                if i % UPDATE_BATCHES_PERIOD == 0:
                    self.optimizers[0].step()
                    self.output_gradient_norms(self.global_step)
                    self.output_gradient_distributions(self.global_step)
                    self.optimizers[0].zero_grad()

            batch_start += cur_batch_size

        if(is_train_mode):
            #update weights
            self.optimizers[0].step()

        return [('loss', loss.div_(len(packed_datapoints.batch_sizes)).item())]

    def train_batch_and_get_metrics(self, batch):
        return self.run_batch_and_get_metrics(batch, is_train_mode=True)

    def test_batch_and_get_metrics(self, batch):
        return self.run_batch_and_get_metrics(batch, is_train_mode=False)

    def intermittent_introspection(self, batch, global_step):
        # Generate a test output for a single sentence
        gt_delta_points = [(0, 0, 0)]
        generated_delta_points = [(0, 0, 0)]
        test_sentence = {}
        # Get one sentence from batch
        test_sentence['datapoints'] = batch['datapoints'][:1, :]
        test_sentence['line_text_integers'] = batch['line_text_integers'][:1, :]
        test_sentence['orig_datapoints_len'] = batch['orig_datapoints_len'][0]
        # Need to resize writer_id to 1D tensor for GeneratorCell invariant shape consistency
        test_sentence['writer_id'] = batch['writer_id'][0].reshape(1,)

        last_hidden_and_cell_states = []
        for _ in range(constants.LSTM_DEPTH):
            last_hidden = torch.zeros(test_sentence['datapoints'].shape[0], constants.LSTM_HIDDEN_SIZE)
            last_cell   = torch.zeros(test_sentence['datapoints'].shape[0], constants.LSTM_HIDDEN_SIZE)

            if torch.cuda.is_available():
                last_hidden, last_cell = last_hidden.cuda(), last_cell.cuda()

            last_hidden_and_cell_states.append((last_hidden, last_cell))
        
        for i in range(test_sentence['orig_datapoints_len']):
            #do forward pass
            letter_id_sequences = test_sentence['line_text_integers']
            writer_ids = test_sentence['writer_id']

            last_hidden_and_cell_states = \
                self.nets[0](writer_ids, letter_id_sequences, last_hidden_and_cell_states)

            #compute loss
            gt = test_sentence['datapoints'][0][i]
            gt_delta_points.append((gt[0], gt[1], gt[2]))
            generated = last_hidden_and_cell_states[-1][0][:, :3]
            
            generated_xy  = generated.narrow(1, 0, 2)
            generated_p   = generated.narrow(1, 2, 1)[0][0]
            generated_p = torch.sigmoid(generated_p)
            # Each generated value is a 2D array
            generated_delta_points.append((generated_xy[0][0], generated_xy[0][1], 1 if generated_p > 0.5 else 0))

        delta_points_to_image(generated_delta_points, constants.INTERMITTENT_OUTPUTS_BASE_DIR, f'output_{global_step}.png')
        delta_points_to_image(gt_delta_points, constants.INTERMITTENT_OUTPUTS_BASE_DIR, f'ground_truth_{global_step}.png')

    @staticmethod
    def generator_loss(generated, gt, writer=None, global_step=0):
        xys  = generated.narrow(1, 0, 2)
        ps   = generated.narrow(1, 2, 1)

        #print(xys)
        #print(ps)

        mse_loss = nn.L1Loss().cuda() if torch.cuda.is_available() else nn.MSELoss()
        bce_loss = nn.BCEWithLogitsLoss().cuda() if torch.cuda.is_available() else nn.BCEWithLogitsLoss()

        mse_loss_val = mse_loss(xys, gt.narrow(1, 0, 2))
        bce_loss_val = bce_loss(ps, gt.narrow(1, 2, 1))
        final_loss_val = mse_loss_val + bce_loss_val

        loss_proportions = {
            'mse': mse_loss_val / final_loss_val * 100.0,
            'bce': bce_loss_val / final_loss_val * 100.0
        }

        if writer != None:
            for loss_type in loss_proportions:
                writer.add_scalar(f'Generator/loss_proportion/{loss_type}', loss_proportions[loss_type], global_step=global_step)

        return final_loss_val

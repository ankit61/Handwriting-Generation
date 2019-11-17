from GeneratorCell import GeneratorCell
from BaseRunner import BaseRunner
from utils import delta_points_to_image, delta_points_to_image_discrete, points_to_image_discrete
import constants
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import constants
import torch
from tqdm import tqdm
from BaseModule import BaseModule
import numpy as np
from HWGANDataset import HWGANDataset

LR = 0.05
MOMENTUM = 0.9
WEIGHT_DECAY = 0
GRADIENT_CLIP_NORM = 100
UPDATE_BATCHES_PERIOD = 20

class GeneratorLoss(BaseModule):
    def __init__(self, writer, model, debug=True):
        super(GeneratorLoss, self).__init__(debug)
        self.writer  = writer
        self.model   = model
        self.xy_loss = nn.MSELoss()
        self.p_loss  = nn.BCEWithLogitsLoss(reduction='mean', 
                            pos_weight=HWGANDataset.pos_weight)

        if torch.cuda.is_available():
            self.xy_loss = self.xy_loss.cuda()
            self.p_loss = self.p_loss.cuda()

    def forward(self, generated, gt, global_step):
        xys  = generated.narrow(1, 0, 2)
        ps   = generated.narrow(1, 2, 1)
        
        print(generated[0].data, gt[0].data)

        loss_vals_weights = {
            'mse': (self.xy_loss(xys, gt.narrow(1, 0, 2)), 1),
            'bce': (self.p_loss(ps, gt.narrow(1, 2, 1)), 100),
            'invariant_regularization': (self.model.invariant.weight.norm(), 0.05)
            #'big_step_penalty': (xys.norm(), 50)
        }
        
        final_loss_val = sum([v[0] * v[1] for k, v in loss_vals_weights.items()])

        if self.writer != None:
            for loss_name, loss_val_weight in loss_vals_weights.items():
                loss_proportion = loss_val_weight[0] * loss_val_weight[1] / final_loss_val
                self.writer.add_scalar(self.__class__.__name__ + f'/loss_proportion/{loss_name}', loss_proportion, global_step=global_step)
                self.writer.add_scalar(self.__class__.__name__ + f'/loss_raw/{loss_name}', loss_val_weight[0], global_step=global_step)

        return final_loss_val

class SupervisedGeneratorRunner(BaseRunner):
    def __init__(self, debug = True):
        model = GeneratorCell()
        optimizer = optim.Adam(model.parameters(), lr=LR)
        loss_fn = GeneratorLoss(None, model)
        self.global_step = 0
        self.force_teach_probability = 1
        
        super(SupervisedGeneratorRunner, self).__init__(models=[model],
            loss_fn=loss_fn, 
            optimizers=[optimizer], best_metric_name='loss', 
            should_minimize_best_metric=True)
        
        self.loss_fn.writer = self.writer
        self.set_gpu_keys(['datapoints', 'writer_id', 'line_text_integers'])

    def run_batch_and_get_metrics(self, batch, is_train_mode):
        #batch['datapoints'].shape -> batch_size x max_seq_len x features_of_element
        packed_datapoints = rnn_utils.pack_padded_sequence(batch['datapoints'],
            batch['orig_datapoints_len'].cpu(), batch_first=True)

        batch_start = 0
        last_hidden_and_cell_states = []

        #initialize last_hidden_and_cell_states
        for _ in range(constants.RNN_DEPTH):
            last_hidden = torch.zeros(batch['datapoints'].shape[0], constants.RNN_HIDDEN_SIZE).uniform_(-1, 1)
            last_cell   = torch.zeros(batch['datapoints'].shape[0], constants.RNN_HIDDEN_SIZE).uniform_(-1, 1)

            if torch.cuda.is_available():
                last_hidden, last_cell = last_hidden.cuda(), last_cell.cuda()

            last_hidden_and_cell_states.append((last_hidden, last_cell))

        loss = 0.0
        self.optimizers[0].zero_grad()

        for i, cur_batch_size in tqdm(enumerate(packed_datapoints.batch_sizes)):
            #do forward pass
            letter_id_sequences = batch['line_text_integers'][:cur_batch_size, :]
            writer_ids = batch['writer_id'][:cur_batch_size]
            gt = packed_datapoints.data[batch_start:batch_start + cur_batch_size, :]

            for j in range(constants.RNN_DEPTH):
                if last_hidden_and_cell_states[j][1] is None:
                    last_hidden_and_cell_states[j] = (last_hidden_and_cell_states[j][0][:cur_batch_size, :], 
                                                        None)
                else:
                    last_hidden_and_cell_states[j] = (last_hidden_and_cell_states[j][0][:cur_batch_size, :],
                                                        last_hidden_and_cell_states[j][1][:cur_batch_size, :])

            if(np.random.rand() < self.force_teach_probability):
                new_hidden = torch.zeros(last_hidden_and_cell_states[-1][0].shape)
                new_hidden[:, 3:] = last_hidden_and_cell_states[-1][0][:, 3:]
                new_hidden[:, :2] = gt.narrow(1, 0, 2)
                new_hidden[:, 2]  = torch.squeeze(gt.narrow(1, 2, 1), axis=1)
                last_hidden_and_cell_states = last_hidden_and_cell_states[:-1] + \
                        [(new_hidden, last_hidden_and_cell_states[-1][1])]

            final_out, last_hidden_and_cell_states = self.nets[0](writer_ids, letter_id_sequences,
                                                    last_hidden_and_cell_states)

            #compute loss

            loss += self.loss_fn(final_out, gt, self.global_step)
            if is_train_mode:
                #calculate gradients but don't update
                loss.backward(retain_graph=(i < len(packed_datapoints.batch_sizes) - 1))
                torch.nn.utils.clip_grad_norm_(self.nets[0].parameters(), GRADIENT_CLIP_NORM)
                self.global_step += 1
                if (i + 1) % UPDATE_BATCHES_PERIOD == 0:
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
        for _ in range(constants.RNN_DEPTH):
            last_hidden = torch.zeros(test_sentence['datapoints'].shape[0], constants.RNN_HIDDEN_SIZE)
            if self.nets[0].rnn_type == 'LSTM':
                last_cell = torch.zeros(test_sentence['datapoints'].shape[0], constants.RNN_HIDDEN_SIZE)
            else:
                last_cell = None

            if torch.cuda.is_available():
                if last_cell is None:
                    last_hidden = last_hidden.cuda()
                else:
                    last_hidden, last_cell = last_hidden.cuda(), last_cell.cuda()

            last_hidden_and_cell_states.append((last_hidden, last_cell))
        
        for i in range(test_sentence['orig_datapoints_len']):
            #do forward pass
            letter_id_sequences = test_sentence['line_text_integers']
            writer_ids = test_sentence['writer_id']

            final_out, last_hidden_and_cell_states = \
                self.nets[0](writer_ids, letter_id_sequences, last_hidden_and_cell_states)

            #compute loss
            gt = test_sentence['datapoints'][0][i]
            gt_delta_points.append((float(gt[0]), float(gt[1]), float(gt[2])))
            generated = final_out

            generated_xy  = generated.narrow(1, 0, 2)
            generated_p   = generated.narrow(1, 2, 1)[0][0]
            generated_p = torch.sigmoid(generated_p)
            # Each generated value is a 2D array
            generated_delta_points.append((float(generated_xy[0][0]), float(generated_xy[0][1]), 1 if generated_p > 0.5 else 0))

        #FIXME: produce this: https://raw.githubusercontent.com/wezteoh/handwriting_generation/master/examples/conditional_generation.png
        delta_points_to_image_discrete(generated_delta_points, constants.INTERMITTENT_OUTPUTS_BASE_DIR, 
            f'output_{global_step}.png', attn_weights=self.nets[0].attn.attn_weights, orig_text='find this yourself!')
        delta_points_to_image_discrete(gt_delta_points, constants.INTERMITTENT_OUTPUTS_BASE_DIR, f'ground_truth_{global_step}.png')

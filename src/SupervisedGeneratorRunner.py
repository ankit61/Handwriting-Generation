from GeneratorCell import GeneratorCell
from BaseRunner import BaseRunner
from utils import points_to_image, attention_output
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

torch.autograd.set_detect_anomaly(True)

LR = 0.05
MOMENTUM = 0.85
WEIGHT_DECAY = 4e-4
GRADIENT_CLIP_NORM = 10
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
        
        #print(generated[0].data, gt[0].data)

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

        return final_loss_val, loss_vals_weights

class SupervisedGeneratorRunner(BaseRunner):
    def __init__(self, debug = True, load_paths = None):
        model = GeneratorCell()
        if torch.cuda.is_available():
            model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        loss_fn = GeneratorLoss(None, model)
        self.global_step = 0
        self.force_teach_probability = 1

        super(SupervisedGeneratorRunner, self).__init__(models=[model],
            loss_fn=loss_fn, 
            optimizers=[optimizer], best_metric_name='loss', 
            should_minimize_best_metric=True, debug=debug, load_paths=load_paths)

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
            last_hidden = torch.zeros(batch['datapoints'].shape[0], constants.RNN_HIDDEN_SIZE)
            last_cell   = torch.zeros(batch['datapoints'].shape[0], constants.RNN_HIDDEN_SIZE)

            if torch.cuda.is_available():
                last_hidden, last_cell = last_hidden.cuda(), last_cell.cuda()

            last_hidden_and_cell_states.append((last_hidden, last_cell))

        last_out = torch.zeros((batch['datapoints'].shape[0], constants.RNN_OUT_SIZE))
        loss = 0.0
        self.optimizers[0].zero_grad()

        batch_size = batch['datapoints'].shape[0]
        last_kappa = torch.zeros((batch_size, constants.ATTENTION_NUM_GAUSSIAN_FUNC))

        for i, cur_batch_size in tqdm(enumerate(packed_datapoints.batch_sizes)):
            #do forward pass
            letter_id_sequences = batch['line_text_integers'][:cur_batch_size, :]
            writer_ids = batch['writer_id'][:cur_batch_size]
            orig_text_lens = batch['orig_line_text_len'][:cur_batch_size]

            for j in range(constants.RNN_DEPTH):
                if last_hidden_and_cell_states[j][1] is None:
                    last_hidden_and_cell_states[j] = (last_hidden_and_cell_states[j][0][:cur_batch_size, :], 
                                                        None)
                else:
                    last_hidden_and_cell_states[j] = (last_hidden_and_cell_states[j][0][:cur_batch_size, :],
                                                        last_hidden_and_cell_states[j][1][:cur_batch_size, :])

            last_out = last_out[:cur_batch_size]

            new_out = last_out.clone().detach()
            new_out = new_out[:cur_batch_size]

            if torch.cuda.is_available():
                new_out = new_out.cuda()

            if(is_train_mode):
                if(i > 0):
                    gt = gt[:cur_batch_size]
                    high_mse_index = (new_out[:, :2] - gt[:, :2]) \
                                        .norm(dim=1) >= constants.XY_PRED_TOLERANCE
                    high_bce_index = ((new_out[:, 2] > constants.SIGMOID_THRESH_P) \
                                        .type(new_out.dtype) - gt[:, 2]).abs() > 0
                    
                    #set gt
                    new_out[high_mse_index, :2] = gt[high_mse_index, :2]
                    new_out[high_bce_index, 2]  = gt[high_bce_index, 2]

            last_out, last_hidden_and_cell_states, last_kappa = self.nets[0](writer_ids, letter_id_sequences, orig_text_lens,
                                                    last_hidden_and_cell_states, new_out, last_kappa[:cur_batch_size, :])
            

            gt = packed_datapoints.data[batch_start:batch_start + cur_batch_size, :]
            
            #compute loss
            cur_loss, _ = self.loss_fn(last_out, gt, self.global_step)
            loss += cur_loss

            batch_start += cur_batch_size

        if(is_train_mode):
            #update weights
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.nets[0].parameters(), GRADIENT_CLIP_NORM)
            self.global_step += 1
            self.output_gradient_norms(self.global_step)
            self.output_gradient_distributions(self.global_step)
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
        test_sentence['orig_line_text_len'] = batch['orig_line_text_len'][0]
        test_sentence['line_text'] = batch['line_text'][0]
        # Need to resize writer_id to 1D tensor for GeneratorCell invariant shape consistency
        test_sentence['writer_id'] = batch['writer_id'][0].reshape(1,)

        last_hidden_and_cell_states = []
        with torch.no_grad():
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

            last_out = torch.zeros((1, constants.RNN_OUT_SIZE))

            attn_weights = torch.zeros(constants.MAX_LINE_TEXT_LENGTH, test_sentence['orig_datapoints_len'])
            last_kappa = torch.zeros((1, constants.ATTENTION_NUM_GAUSSIAN_FUNC))

            for i in range(test_sentence['orig_datapoints_len']):
                #do forward pass
                letter_id_sequences = test_sentence['line_text_integers']
                writer_ids = test_sentence['writer_id']
                gt = test_sentence['datapoints'][0][i]
                orig_text_lens = test_sentence['orig_line_text_len'].unsqueeze(0)

                # if(np.random.rand() < self.force_teach_probability):
                #     new_out = torch.zeros(last_out.shape)
                #     if torch.cuda.is_available():
                #         new_out = new_out.cuda()
                #     new_out[0, :2] = gt[0:2]
                #     new_out[0, 2]  = gt[2]
                # else:
                new_out = last_out

                last_out, last_hidden_and_cell_states, last_kappa = \
                    self.nets[0](writer_ids, letter_id_sequences, orig_text_lens, last_hidden_and_cell_states, new_out, last_kappa)

                attn_weights[:, i] = self.nets[0].attn.get_attn_weights()

                #compute loss
                gt_delta_points.append((float(gt[0]), float(gt[1]), float(gt[2])))
                generated = last_out

                generated_xy  = generated.narrow(1, 0, 2)
                generated_p   = generated.narrow(1, 2, 1)[0][0]
                generated_p = torch.sigmoid(generated_p)
                # Each generated value is a 2D array
                generated_delta_points.append((float(generated_xy[0][0]), float(generated_xy[0][1]), 1 if generated_p > constants.SIGMOID_THRESH_P else 0))

            points_plot = points_to_image(generated_delta_points, ground_truth_points=gt_delta_points, delta_points=True)
                        #, attn_weights=self.nets[0].attn.attn_weights, orig_text='find this yourself!')
            self.writer.add_figure(f'{self.name}/intermittent_output', points_plot, global_step=self.global_step)

            attn_heatmap = attention_output(attn_weights.data, generated_delta_points, test_sentence['line_text'][:test_sentence['orig_line_text_len']]
                , delta_points=True)
            self.writer.add_figure(f'{self.name}/attention_heatmap', attn_heatmap, global_step=self.global_step)

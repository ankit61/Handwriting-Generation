import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from abc import abstractmethod, ABCMeta
import tensorboardX
import utils
import time
import constants
import os
from numpy import sign
import torch.optim.lr_scheduler as lr_scheduler

LR_DECAY_STEP_SIZE  = 10
LR_DECAY_FACTOR     = 0.5

class BaseRunner(metaclass=ABCMeta):
    #inspired by https://github.com/pytorch/examples/blob/master/imagenet/main.py

    def __init__(self, models, loss_fn, optimizers, best_metric_name,
        should_minimize_best_metric, debug = True, introspect = True):
        
        assert type(models) == type([]), 'models must be a list'
        assert type(optimizers) == type([]), 'optimizers must be a list'

        self.writer = tensorboardX.SummaryWriter(constants.TENSORBOARDX_BASE_DIR)
        self.nets   = models
        self.name   = self.__class__.__name__
        self.debug  = debug
        self.introspect = introspect
        self.best_metric_name = best_metric_name
        self.best_compare = -1 if should_minimize_best_metric else 1
        self.best_metric_val = - self.best_compare * 100000
        self.best_meter = utils.AverageMeter('best_metric')
        self.loss_fn = loss_fn
        self.optimizers = optimizers
        self.keys_for_gpu = None
        self.lr_schedulers = \
            [lr_scheduler.StepLR(optimizers[i], LR_DECAY_STEP_SIZE, LR_DECAY_FACTOR) 
                for i in range(len(self.optimizers))]

        if(torch.cuda.is_available()):
            for i in range(len(self.nets)):
                self.nets[i] = self.nets[i].cuda()

    def output_weight_distribution(self, name_prefix="training_weights"):
        if not self.introspect:
            return

        for net in self.nets:
            for param_name, param_val in net.named_parameters():
                param_distribution_tag = f'{net.__class__.__name__}/{name_prefix}/{param_name}'
                self.writer.add_histogram(param_distribution_tag, param_val)
    
    def output_gradient_distributions(self, global_step, name_prefix="training_gradients"):
        if not self.introspect:
            return

        for net in self.nets:
            for param_name, param in net.named_parameters():
                param_distribution_tag = f'{net.__class__.__name__}/{name_prefix}/{param_name}'
                self.writer.add_histogram(param_distribution_tag, param.grad, global_step=global_step)
    
    def output_gradient_norms(self, global_step, name_prefix="training_gradient_norms"):
        if not self.introspect:
            return

        for net in self.nets:
            for param_name, param in net.named_parameters():
                param_distribution_tag = f'{net.__class__.__name__}/{name_prefix}/{param_name}'
                self.writer.add_scalar(param_distribution_tag, torch.norm(param.grad), global_step=global_step)
    
    def output_weight_norms(self, global_step, name_prefix="training_weight_norms"):
        if not self.introspect:
            return

        for net in self.nets:
            for param_name, param in net.named_parameters():
                param_distribution_tag = f'{net.__class__.__name__}/{name_prefix}/{param_name}'
                self.writer.add_scalar(param_distribution_tag, torch.norm(param), global_step=global_step)

    def set_gpu_keys(self, keys):
        self.keys_for_gpu = keys

    def run(self, data_loader, prefix, epoch, metrics_calc):
        batch_time_meter = utils.AverageMeter('Time')
        data_time_meter  = utils.AverageMeter('Data')
        other_meters = []
        
        progress_display_made = False
        start_time = time.time()

        for i, batch in enumerate(data_loader):
            batch_number = epoch * len(data_loader) + i + 1
            data_time_meter.update(time.time() - start_time)

            self.output_weight_norms(epoch)

            if batch_number % constants.INTERMITTENT_OUTPUT_FREQ == 0:
                self.intermittent_introspection(batch, batch_number)

            #transfer from CPU -> GPU asynchronously if at all
            if torch.cuda.is_available():
                if type(batch) != type([]) and type(batch) != type({}):
                    batch = batch.cuda(non_blocking=True)
                elif type(batch) == type([]):
                    for j in range(len(batch)):
                        batch[j] = batch[j].cuda(non_blocking=True)
                else: #type(batch) == type({})
                    for key in batch.keys():
                        if self.keys_for_gpu is None or key in self.keys_for_gpu:
                            batch[key] = batch[key].cuda(non_blocking=True)

            metrics = metrics_calc(batch)
            # loss.backward is called in metrics_calc
            if metrics is not None:
                for j, (metric_name, metric_val) in enumerate(metrics):
                    self.writer.add_scalar(os.path.join(self.name, prefix + '_' + 
                        metric_name), metric_val, batch_number)

                    if not progress_display_made:
                        other_meters.append(utils.AverageMeter(metric_name))
                        other_meters[j].update(metric_val)
                    else:
                        other_meters[j].update(metric_val)

                if not progress_display_made:
                    progress = utils.ProgressMeter(len(data_loader), other_meters + \
                        [batch_time_meter, data_time_meter], prefix=prefix)
                    progress_display_made = True
            elif not progress_display_made:
                progress = utils.ProgressMeter(len(data_loader), [batch_time_meter, data_time_meter], prefix=prefix)

            batch_time_meter.update(time.time() - start_time)
            start_time = time.time()

            if i % constants.PRINT_FREQ == 0:
                progress.display(i, epoch)

    def train(self, train_loader, epochs, val_loader = None):
        self.output_weight_distribution("weight_initializations")

        for i in range(len(self.nets)):
            self.nets[i].train()

        for epoch in range(epochs):
            self.run(train_loader, 'train', epoch, self.train_batch_and_get_metrics)

            for i in range(len(self.lr_schedulers)):
                self.lr_schedulers[i].step()
                
            if val_loader is not None:
                self.test(val_loader, validate=True)
                if(sign(self.best_meter.avg - self.best_metric_val) == self.best_compare):
                    for i in range(len(self.nets)):
                        torch.save({
                            'arch': self.nets[i].__class__.__name__,
                            'state_dict': self.nets[i].state_dict(),
                            'best_metric_val': self.best_meter.avg,
                            'best_metric_name': self.best_metric_name
                            }, os.path.join(constants.MODELS_BASE_DIR,
                                self.nets[i].__class__.__name__ + '_' + \
                                'checkpoint_' + str(epoch + 1) + '.pth')
                        )
                        self.best_metric_val = self.best_meter.avg
                self.best_meter.reset()

        self.output_weight_distribution("final_weights")

    def test(self, test_loader, validate=False):
        for i in range(len(self.nets)):
            self.nets[i].eval()

        with torch.no_grad():
            if validate:
                self.run(test_loader, 'test', 1, self.validate_batch_and_get_metrics)
            else:
                self.run(test_loader, 'test', 1, self.test_batch_and_get_metrics)

    def validate_batch_and_get_metrics(self, batch):
        metrics = self.test_batch_and_get_metrics(batch)
        did_find_name = False
        for (metric_name, metric_val) in metrics:
            if metric_name == self.best_metric_name:
                self.best_meter.update(metric_val)
                did_find_name = True
                break

        if not did_find_name:
            raise Exception('''Invalid best_metric_name set - 
                best_metric_name must be one of metrics
                best_metric_name: {}
                metric names: {}'''.format(self.best_metric_name, \
                [x[0] for x in metrics])
            )
        return metrics

    @abstractmethod
    def train_batch_and_get_metrics(self, batch):
        '''Perform forward and backward pass here. Also perform the actual 
            update by doing optimizer.step() (remember to do 
            optimizer.zero_grad()).  Finally, use a learning rate scheduler
            (default choice can be torch.optim.lr_scheduler.StepLR)
           
            Return: metrics - [(metric_name, metric_val (should be scalar))]
        '''
        return

    @abstractmethod
    def test_batch_and_get_metrics(self, batch):
        '''Perform forward pass here.
           
            Return: metrics - [(metric_name, metric_val (should be scalar))]'''
        return

    def intermittent_introspection(self, batch):
        '''Perform any intermittent statistics / test output introspection here
            * This function will be called intermittently throughout the training process

            Return: NoneType
        '''
        raise NotImplementedError()

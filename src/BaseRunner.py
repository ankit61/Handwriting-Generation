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

class BaseRunner(metaclass=ABCMeta):
    #inspired by https://github.com/pytorch/examples/blob/master/imagenet/main.py

    def __init__(self, models, loss_fn, optimizers, best_metric_name,
        should_minimize_best_metric, debug = True):
        
        assert type(models) == type([]), 'models must be a list'
        assert type(optimizers) == type([]), 'optimizers must be a list'

        self.writer = tensorboardX.SummaryWriter(constants.TENSORBOARDX_BASE_DIR)
        self.nets   = models
        self.name   = self.__class__.__name__
        self.debug  = debug
        self.best_metric_name = best_metric_name
        self.best_compare = -1 if should_minimize_best_metric else 1
        self.best_metric_val = - self.best_compare * 100000
        self.best_meter = utils.AverageMeter('best_metric')
        self.loss_fn = loss_fn
        self.optimizers = optimizers

    def run(self, data_loader, prefix, epoch, metrics_calc):
        batch_time_meter = utils.AverageMeter('Time')
        data_time_meter  = utils.AverageMeter('Data')
        other_meters = []
        
        progress_display_made = False
        start_time = time.time()
        for i, batch in enumerate(data_loader):
            data_time_meter.update(time.time() - start_time)

            #transfer from CPU -> GPU asynchronously if at all
            if torch.cuda.is_available():
                if type(batch) != type([]):
                    batch = batch.cuda(non_blocking=True)
                else:
                    for j in range(len(batch)):
                        batch[j] = batch[j].cuda(non_blocking=True)

            metrics = metrics_calc(batch)
            if metrics is not None:
                for j, (metric_name, metric_val) in enumerate(metrics):
                    self.writer.add_scalar(os.path.join(self.name, prefix + '_' + 
                        metric_name), metric_val, epoch * i + i)

                    if not progress_display_made:
                        other_meters.append(utils.AverageMeter(metric_name))
                        other_meters[j].update(metric_val)
                    else:
                        other_meters[j].update(metric_val)

                if not progress_display_made:
                    progress = utils.ProgressMeter(len(data_loader), other_meters + \
                        [batch_time_meter, data_time_meter], prefix=prefix)
                    progress_display_made = True

            batch_time_meter.update(time.time() - start_time)
            start_time = time.time()

            if i % constants.PRINT_FREQ == 0:
                progress.display(i, epoch)

    def train(self, train_loader, epochs, val_loader = None):
        for i in range(len(self.nets)):
            self.nets[i].train()

        for epoch in range(epochs):
            self.run(train_loader, 'train', epoch, self.train_batch_and_get_metrics)

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
                                'checkpoint_' + str(epoch + 1) + '.pth')
                        )
                        self.best_metric_val = self.best_meter.avg
                self.best_meter.reset()

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
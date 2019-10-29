import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from abc import abstractmethod, ABCMeta
import tensorboardX
import utils
import time
import os

class BaseRunner:
    #inspired by https://github.com/pytorch/examples/blob/master/imagenet/main.py
    __metaclass__ = ABCMeta
    def __init__(self, name, models, loss_fn, optimizers, debug = True):
        assert type(models) == type([]), 'models must be a list'
        assert type(optimizers) == type([]), 'optimizers must be a list'

        self.writer = tensorboardX.SummaryWriter(TENSORBOARDX_BASE_DIR)
        self.nets   = models
        self.name   = name
        self.debug  = debug
        self.best_metric = None
        self.loss_fn = loss_fn
        self.optimizers = optimizers

    def run(self, data_loader, prefix, epoch, metrics_calc):
        #TODO: Find a good way to add loss in progress meter
        #loss_meter = utils.AverageMeter('Loss', ':.4e')
        batch_time_meter = utils.AverageMeter('Time', ':6.3f')
        data_time_meter  = utils.AverageMeter('Data', ':6.3f')
        utils.ProgressMeter(len(train_loader), [#loss_meter, 
            batch_time_meter, data_time_meter], prefix=prefix)

        start_time = time.time()
        for i, batch in enumerate(data_loader):
            data_time_meter.update(time.time() - start_time)

            #transfer from CPU -> GPU asynchronously
            if type(batch) != type(tuple):
                batch.cuda(non_blocking=True)
            else:
                for j in range(len(batch)):
                    batch[j].cuda(non_blocking=True)

            metrics = metrics_calc(batch)
            if metrics is not None:
                for (metric_name, metric_val) in metrics:                    
                    writer.add_scalar(os.path.join(self.name, prefix + '_' + 
                        metric_name), metric_val, epoch * i + i)

            batch_time_meter.update(time.time() - start_time)
            start_time = time.time()

            if i % constants.PRINT_FREQ:
                progress.display(i)

    def train(self, train_loader, epochs, val_loader = None):
        assert train_loader.pin_memory, 'set pin_memory to true for faster data transfer'
        assert train_loader.dataset.pin_memory, 'set pin_memory to true for faster data transfer'

        for i in range(len(self.nets)):
            self.nets[i].train()

        for epoch in epochs:
            self.run(train_loader, 'train', epoch, self.train_batch_and_get_metrics)
            
            if val_loader is not None:
                test(val_loader, save_best=True)

    def test(self, test_loader, save_best=False):
        assert train_loader.pin_memory, 'set pin_memory to true for faster data transfer'
        assert train_loader.dataset.pin_memory, 'set pin_memory to true for faster data transfer'

        for i in range(len(self.nets)):
            self.nets[i].eval()

        #make sure to take care of save
        with torch.no_grad():
            self.run(test_loader, 'test', 1, self.test_batch_and_get_metrics)

    @abstractmethod
    def train_batch_and_get_metrics(self, batch):
        '''Perform forward and backward pass here. Also perform the actual 
           update by doing optimizer.step() (remember to do 
           optimizer.zero_grad()).  Finally, use a learning rate scheduler
           (default choice can be torch.optim.lr_scheduler.StepLR)
           
            Return: metrics - [(metric_name, metric_val (should be scalar))]
        '''
        raise NotImplementedError()

        @abstractmethod
    def test_batch_and_get_metrics(self, batch):
        '''Perform forward pass here.
           
            Return: metrics - [(metric_name, metric_val (should be scalar))]
        '''
        raise NotImplementedError()

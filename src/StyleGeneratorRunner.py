import torch
import torch.nn as nn
import tensorboardX
import torchvision.models as models
from BaseRunner import BaseRunner
from StyleGeneratorNetwork import StyleGeneratorNetwork


class StyleGeneratorRunner(BaseRunner):
    def __int__(self, models, loss_fn, optimizers, best_metric_name = 'acc1',
        should_minimize_best_metric = False, debug = True):

        net = StyleGeneratorNetwork()
        loss_fn = nn.MSELoss()

        if torch.cuda.is_available():
            torch.cuda.set_device('cuda:0')
            net = net.cuda()
            loss_fn = loss_fn.cuda()

        optimizer = torch.optim.SGD(net.parameters(), 0.01,
                                momentum=0.9,
                                weight_decay=5e-4)
        ACCURACY_THRESHOLD = 0.05


        super(StyleGeneratorRunner, self).__init__([net], loss_fn, [optimizer],
            best_metric_name, should_minimize_best_metric, debug)
        
    def train_batch_and_get_metrics(self, batch):
        # batch[0] is input
        # batch[1] is ground_truths
        out = self.nets[0](batch[0])
        loss = self.loss_fn(out, batch[1])
        acc1 = self.accuracy(out, batch[1])
        
        self.optimizers[0].zero_grad()
        loss.backward()
        self.optimizers[0].step()

        return [('loss', loss.mean().item()), ('acc1', acc1)]

    def test_batch_and_get_metrics(self, batch):
        # batch[0] is input
        # batch[1] is ground_truths
        out = self.nets[0](batch[0])
        loss = self.loss_fn(out, batch[1])
        acc1 = self.accuracy(out, batch[1])

        return [('loss', loss.mean().item()), ('acc1', acc1)]

    def accuracy(self, network_outputs, ground_truths):
        assert len(network_outputs) == len(ground_truths)
        good_outputs = 0
        for i in range(len(network_outputs)):
            if self.loss_fn(network_outputs[i], ground_truths[i]) <= self.ACCURACY_THRESHOLD:
                good_outputs += 1
        return 1.0 * good_outputs / len(network_outputs)


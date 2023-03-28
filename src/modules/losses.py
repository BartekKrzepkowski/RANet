import torch

from src.modules.metrics import acc_metric
from src.modules.regularizers import FisherPenaly
from src.utils import common


class ClassificationLoss(torch.nn.Module):
    def __init__(self, criterion_name):
        super().__init__()
        self.criterion = common.LOSS_NAME_MAP[criterion_name]()

    def forward(self, y_pred, y_true, postfix=''):
        loss = self.criterion(y_pred, y_true)
        acc = acc_metric(y_pred, y_true)
        evaluators = {
            f'loss{postfix}': loss.item(),
            f'acc{postfix}': acc
        }
        return loss, evaluators


class FisherPenaltyLoss(torch.nn.Module):
    def __init__(self, model, general_criterion_name, num_classes, whether_record_trace=False, fpw=0.0):
        super().__init__()
        self.criterion = ClassificationLoss(general_criterion_name)
        self.regularizer = FisherPenaly(model, common.LOSS_NAME_MAP[general_criterion_name](), num_classes)
        self.whether_record_trace = whether_record_trace
        self.fpw = fpw
        #przygotowanie do logowania co n kroków
        self.overall_trace_buffer = None
        self.traces = None

    def forward(self, y_pred, y_true):
        traces = {}
        loss, evaluators = self.criterion(y_pred, y_true)
        if self.whether_record_trace:# and self.regularizer.model.training:
            overall_trace, traces = self.regularizer(y_pred)
            evaluators['overall_trace'] = overall_trace.item()
            if self.fpw > 0:
                loss += self.fpw * overall_trace
        return loss, evaluators, traces

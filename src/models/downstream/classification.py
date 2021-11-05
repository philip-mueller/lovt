from typing import List

import torch
from torch.nn import functional as F
from torch import nn

from data.datasets.chexpert.chexpert_dataset import chexpert_labels


class ClassificationHead(nn.Module):
    def __init__(self, d_y: int, classifier_task: str = 'chexpert_binary',
                 d_hidden=512, dropout_prob=0.2, nonlinear=False, dataset_stats=None):
        super(ClassificationHead, self).__init__()
        if classifier_task == 'chexpert_binary':
            classifier_loss = MultiTaskBinaryClassifierLoss(chexpert_labels())
            self.labels_name = 'chexpert_bin_labels'
            self.labels_converter = None
        elif classifier_task == 'chexpert_binary_weighted':
            assert dataset_stats is not None
            classifier_loss = MultiTaskBinaryClassifierLoss(chexpert_labels(),
                                                            pos_weights=dataset_stats.get('chexpert_bin_pos_weights'))
            self.labels_name = 'chexpert_bin_labels'
            self.labels_converter = None
        else:
            raise ValueError(classifier_task)

        self.nonlinear = nonlinear
        if nonlinear:
            assert d_hidden is not None
            self.project = nn.Linear(d_y, d_hidden, bias=False)
            self.relu = nn.ReLU(inplace=True)
            self.bn = nn.BatchNorm1d(d_hidden)
        else:
            d_hidden = d_y

        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(d_hidden, classifier_loss.num_logits, bias=True)
        self.classifier_loss = classifier_loss
        self.num_labels = len(self.classifier_loss.tasks)

    def get_labels(self, batch):
        labels = batch[self.labels_name]
        if self.labels_converter:
            labels = self.labels_converter(labels)
        return labels

    def forward(self, yg, labels=None, return_probs=True):
        """

        :param x: (B x d_y)
        :return: probs, labels, loss
        """
        x = yg
        if self.nonlinear:
            x = self.dropout(x)
            x = self.project(x)
            x = self.bn(x)
            x = self.relu(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return self.classifier_loss(x, labels=labels, return_probs=return_probs)


class MultiTaskBinaryClassifierLoss(nn.Module):
    def __init__(self, tasks: List[str], pos_weights: dict = None):
        super(MultiTaskBinaryClassifierLoss, self).__init__()
        self.num_logits = len(tasks)
        self.tasks = tasks
        if pos_weights is not None:
            self.register_buffer('pos_weights', torch.tensor([pos_weights[task] for task in tasks]))
        else:
            self.pos_weights = None

    def stacked_to_dict(self, stacked_tensor: torch.Tensor) -> dict:
        return {task: stacked_tensor[:, i] for i, task in enumerate(self.tasks)}

    def forward(self, logits, labels=None, return_probs=True):
        """

        :param logits: B x N_tasks
        :param labels: dict { task_name: (B) } or B x N_tasks
        :param return_probs: If True return the prediction as probs if false return predictions as logits
        :return:
        """

        if labels is not None:
            # B x N_tasks
            if isinstance(labels, dict):
                labels = torch.stack([labels[task] for task in self.tasks], dim=-1)
            else:
                assert isinstance(labels, torch.Tensor)
            loss = F.binary_cross_entropy_with_logits(logits, labels.type_as(logits), pos_weight=self.pos_weights)
        predictions = torch.sigmoid(logits) if return_probs else logits

        if labels is not None:
            return predictions, labels, loss
        else:
            return predictions


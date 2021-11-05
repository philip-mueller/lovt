from dataclasses import dataclass

import torch
from pytorch_lightning import Callback, Trainer, LightningModule
from pytorch_lightning.utilities import move_data_to_device
from torchmetrics import Accuracy, AUROC, F1

from models.downstream.classification import ClassificationHead
from models.pretraining.bimodal_alignment_model import BiModalModelRepresentationLearner


@dataclass
class OnlineEvaluatorConfig:
    task: str = 'chexpert_binary'
    source_modality: str = 'a'  # 'a', 'b', 'a&b'

    nonlinear: bool = False  # most literature used False
    d_hidden: int = 512  # only relevant if nonlinear = True
    dropout_prob: float = 0.2


def instantiate_online_evaluator(config: OnlineEvaluatorConfig, name) -> Callback:
    if config.task in ('chexpert_binary',):
        return OnlineClassificationEvaluator(config, name)
    else:
        raise ValueError(f'Unsupported online evaluation task {config.task}')


class OnlineClassificationEvaluator(Callback):
    def __init__(self, config: OnlineEvaluatorConfig, name: str):
        super(OnlineClassificationEvaluator, self).__init__()
        self.config = config
        self.name = name

    def on_pretrain_routine_start(self, trainer: Trainer, pl_module: BiModalModelRepresentationLearner) -> None:
        assert isinstance(pl_module, BiModalModelRepresentationLearner)
        assert not hasattr(pl_module, self.name)

        prediction_head = ClassificationHead(
            classifier_task=self.config.task,
            d_y=pl_module.get_representation_dim(modality=self.config.source_modality, representation_type='global'),
            d_hidden=self.config.d_hidden,
            dropout_prob=self.config.dropout_prob,
            nonlinear=self.config.nonlinear
        )

        prediction_head.train_acc_metric = Accuracy()
        prediction_head.val_acc_metric = Accuracy(compute_on_step=False)
        prediction_head.val_auroc_metric = AUROC(compute_on_step=False, num_classes=prediction_head.num_labels)
        prediction_head = prediction_head.to(pl_module.device)

        self.optimizer = torch.optim.Adam(prediction_head.parameters(), lr=1e-4)

        setattr(pl_module, self.name, prediction_head)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        assert isinstance(pl_module, BiModalModelRepresentationLearner)
        batch = move_data_to_device(batch, pl_module.device)
        prediction_head = getattr(pl_module, self.name)
        labels = prediction_head.get_labels(batch)

        with torch.no_grad():
            representations = pl_module.get_representation(batch,
                                                           modality=self.config.source_modality,
                                                           representation_type='global')
        representations = representations.detach()  # (B x d)

        # forward pass
        prediction_head = getattr(pl_module, self.name)
        probs, labels, loss = prediction_head(representations, labels)

        # update finetune weights
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # log metrics
        prediction_head.train_acc_metric(probs, labels)
        pl_module.log(f'train/online_{self.name}_acc', prediction_head.train_acc_metric, on_step=True, on_epoch=False)
        pl_module.log(f'train/online_{self.name}_loss', loss, on_step=True, on_epoch=False)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        batch = move_data_to_device(batch, pl_module.device)
        prediction_head = getattr(pl_module, self.name)
        labels = prediction_head.get_labels(batch)

        with torch.no_grad():
            representations = pl_module.get_representation(batch,
                                                           modality=self.config.source_modality,
                                                           representation_type='global',
                                                           val=True)
        representations = representations.detach()  # (B x d)

        # forward pass
        prediction_head = getattr(pl_module, self.name)
        probs, labels, loss = prediction_head(representations, labels)

        # log metrics
        prediction_head.val_acc_metric(probs, labels)
        prediction_head.val_auroc_metric(probs, labels)
        pl_module.log(f'val/online_{self.name}_acc', prediction_head.val_acc_metric)
        pl_module.log(f'val/online_{self.name}_auroc', prediction_head.val_auroc_metric)
        pl_module.log(f'val/online_{self.name}_loss', loss)

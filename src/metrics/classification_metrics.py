import torch
from pytorch_lightning.metrics import Metric


class TopKAccuracy(Metric):
    """
    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch
    """
    def __init__(self, topk=(1,), name='top_{k}_acc', dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.topk = topk
        self.maxk = max(topk)
        self.names = [name.format(k=k) for k in topk]
        self.add_state("correct", default=torch.zeros(len(topk)), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor, target: torch.Tensor):
        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        maxk = min(self.maxk, int(logits.size(1)))
        _, y_pred = logits.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [1, B] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        for i, k in enumerate(self.topk):
            k = min(k, maxk)
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(
                -1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0,
                                                                                        keepdim=True)  # [kB] -> [1]
            self.correct[i, None] += tot_correct_topk
        self.total += target.size(0)  # batch size

    def compute(self):
        topk_accuracies = self.correct.float() / self.total  # (len(topk))
        return {name: acc for name, acc in zip(self.names, topk_accuracies)}
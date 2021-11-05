from torch import nn
from transformers.activations import get_activation
from models.components.utils import get_norm_layer


class SequenceMLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, d_hidden: int = None,
                 bias=True, dropout_prob=0.,
                 norm='batch', norm_before_act=True, act='relu', channels_last=True):
        super(SequenceMLP, self).__init__()
        self.norm_before_act = norm_before_act
        self.channels_last = channels_last

        if d_hidden is None:
            d_hidden = d_out
        self.projection_before = nn.Conv1d(d_in, d_hidden, kernel_size=1, bias=bias)
        if act is not None:
            self.act = get_activation(act)
            self.projection_after = nn.Conv1d(d_hidden, d_out, kernel_size=1, bias=bias)
        else:
            self.act = lambda x: x
            self.projection_after = lambda x: x
        self.dropout = nn.Dropout(dropout_prob)
        self.norm = get_norm_layer(norm, d_hidden) if norm_before_act else get_norm_layer(norm, d_out)

        nn.init.kaiming_normal_(self.projection_before.weight, mode='fan_out', nonlinearity=act)
        nn.init.kaiming_normal_(self.projection_after.weight, mode='fan_out', nonlinearity='linear')
        if bias:
            nn.init.zeros_(self.projection_before.bias)
            nn.init.zeros_(self.projection_after.bias)

    def forward(self, x):
        """

        :param x: (B x N x d)
        :return: (B x N x d)
        """
        assert x.ndim == 3
        if self.channels_last:
            x = x.transpose(-1, -2)  # (B x d x N), transpose for conv
        x = self.projection_before(x)
        if self.norm_before_act:
            x = self._apply_norm(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.projection_after(x)

        if not self.norm_before_act:
            x = self._apply_norm(x)
        if self.channels_last:
            x = x.transpose(-1, -2)  # (B x N x d), transpose back from conv
        return x

    def _apply_norm(self, x):
        if isinstance(self.norm, nn.BatchNorm1d):
            return self.norm(x)
        else:  # other norms work on the last dim, but conv uses dim -2 as the channel dim
            return self.norm(x.transpose(-1, -2)).transpose(-1, -2)


class MLP(nn.Module):
    """
    Nonlinear porjections model
    See also:
        - https://github.com/lucidrains/pixel-level-contrastive-learning/blob/0c60e93df73b0ec351f2104839c4d4748f3d38a2/pixel_level_contrastive_learning/pixel_level_contrastive_learning.py#L89
          -> MLP
          or https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/1bec12200ff60b9b58277b40d9232933e1aa9dbe/pl_bolts/models/self_supervised/byol/models.py#L24
    """
    def __init__(self, d_in: int, d_out: int, d_hidden: int = None,
                 bias=True, dropout_prob=0.,
                 norm='batch', norm_before_act=True, act='relu',):
        super(MLP, self).__init__()
        self.norm_before_act = norm_before_act

        if d_hidden is None:
            d_hidden = d_out
        self.projection_before = nn.Linear(d_in, d_hidden, bias=bias)
        if act is not None:
            self.act = get_activation(act)
            self.projection_after = nn.Linear(d_hidden, d_out, bias=bias)
        else:
            self.act = lambda x: x
            self.projection_after = lambda x: x
        self.dropout = nn.Dropout(dropout_prob)
        self.norm = get_norm_layer(norm, d_hidden) if norm_before_act else get_norm_layer(norm, d_out)

        nn.init.kaiming_normal_(self.projection_before.weight, mode='fan_out', nonlinearity=act)
        nn.init.kaiming_normal_(self.projection_after.weight, mode='fan_out', nonlinearity='linear')
        if bias:
            nn.init.zeros_(self.projection_before.bias)
            nn.init.zeros_(self.projection_after.bias)

    def forward(self, x):
        """

        :param x: (B x d)
        :return: (B x d)
        """
        assert x.ndim == 2

        x = self.projection_before(x)
        if self.norm_before_act:
            x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.projection_after(x)
        if not self.norm_before_act:
            x = self.norm(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionNet(nn.Module):

    def __init__(self, attention_size):
        super().__init__()

        self.attention_size = attention_size

        self.attention = nn.Linear(self.attention_size, self.attention_size)

    def forward(self, x, z=None):
        """

        :param x: attention base
        :param z: if None, self attention is calculated, else z is attended to by x
        :return:
        """

        return (x if z is None else z) * F.softmax(self.attention(x), dim=-1)


class WavenetLayer(nn.Module):
    def __init__(self, ch_residual=32, ch_dilation=32, ch_skip=64, kernel_size=3, dilation=2, bias=True,
                 causal=True) -> None:
        super().__init__()

        """Parameters"""
        self.ch_residual = ch_residual
        self.ch_dilation = ch_dilation
        self.ch_skip = ch_skip
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.bias = bias
        self.causal = causal


        # self.causal=False


        """Padding"""
        # source: https://github.com/r9y9/wavenet_vocoder/blob/master/wavenet_vocoder/modules.py#L80
        # see also: https://pytorch.org/docs/stable/nn.html#conv1d and
        # https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338

        # no future time stamps available
        if self.causal:
            self.padding = (self.kernel_size - 1) * self.dilation
        else:
            self.padding = ((self.kernel_size - 1) * self.dilation) // 2

        """Layers"""

        # dilated convolutions for the gated activation unit
        self.filter_conv = nn.Conv1d(in_channels=self.ch_residual, out_channels=self.ch_dilation,
                                     kernel_size=self.kernel_size, padding=self.padding, dilation=self.dilation,
                                     bias=self.bias)
        self.gate_conv = nn.Conv1d(in_channels=self.ch_residual, out_channels=self.ch_dilation,
                                   kernel_size=self.kernel_size, padding=self.padding, dilation=self.dilation,
                                   bias=self.bias)

        self.bn = nn.BatchNorm1d(self.ch_dilation)
        self.skip_bn = nn.BatchNorm1d(self.ch_skip)
        self.residual_bn = nn.BatchNorm1d(self.ch_residual)

        # 1x1 convolutions for residual and skip connections
        self.residual_conv = nn.Conv1d(in_channels=self.ch_dilation, out_channels=self.ch_residual, kernel_size=1,
                                       bias=self.bias)
        self.skip_conv = nn.Conv1d(in_channels=self.ch_dilation, out_channels=self.ch_skip, kernel_size=1,
                                   bias=self.bias)

    def _make_causal(self, x):
        """
        Makes the output of a (convolutional) layer causal.

        Sources: - https://github.com/pytorch/pytorch/issues/1333
                 - https://github.com/r9y9/wavenet_vocoder/blob/master/wavenet_vocoder/modules.py#L136

        :param x:
        :return:
        """
        return x if not self.causal else x[..., :-self.padding]

    def forward(self, x):
        # save input for the residual connection
        # input = x.clone()

        # gated activation unit
        filter = torch.tanh(self.filter_conv(x))
        gate = torch.sigmoid(self.gate_conv(x))
        # gated_activation_out = self.bn(F.relu(self._make_causal(filter) * self._make_causal(gate)))
        gated_activation_out = self._make_causal(filter) * self._make_causal(gate)

        # skip connection
        # skip = self.skip_bn(F.relu(self.skip_conv(gated_activation_out)))
        skip = self.skip_conv(gated_activation_out)

        # residual connection
        residual_output = self.residual_conv(gated_activation_out) + x

        return residual_output, skip


class WavenetBlock(nn.Module):

    def __init__(self, num_layers=2, ch_residual=32, ch_dilation=32, ch_skip=64, kernel_size=3, bias=True,
                 causal=True) -> None:
        super().__init__()

        """Parameters"""
        self.num_layers = num_layers
        self.ch_residual = ch_residual
        self.ch_dilation = ch_dilation
        self.ch_skip = ch_skip
        self.kernel_size = kernel_size
        self.bias = bias
        self.causal = causal

        self.layers = nn.ModuleList([
            WavenetLayer(self.ch_residual, self.ch_dilation, self.ch_skip, self.kernel_size, dilation=(2 ** i),
                         bias=self.bias, causal=self.causal) for i in range(self.num_layers)])

    def forward(self, residual):
        skip_outputs = []
        for layer in self.layers:
            residual, skip = layer(residual)
            skip_outputs.append(skip)

        return residual, torch.stack(skip_outputs)


class WaveNet(nn.Module):

    def __init__(self, num_blocks=2, num_layers=2, num_classes=4, output_len=16, ch_start=16, ch_residual=32,
                 ch_dilation=32, ch_skip=64, ch_end=32, kernel_size=3, bias=True, causal=True) -> None:
        super().__init__()

        """Parameters"""
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.output_len = output_len

        self.ch_start = ch_start
        self.ch_residual = ch_residual
        self.ch_dilation = ch_dilation
        self.ch_skip = ch_skip
        self.ch_end = ch_end
        self.kernel_size = kernel_size
        self.bias = bias
        self.causal = causal

        """Layers"""

        # 1x1 convolution to create channels
        self.start_conv = nn.Conv1d(in_channels=self.ch_start, out_channels=self.ch_residual, kernel_size=1,
                                    bias=self.bias)

        # wavenet blocks
        self.blocks = nn.ModuleList([WavenetBlock(self.num_layers, self.ch_residual, self.ch_dilation, self.ch_skip,
                                                  self.kernel_size, self.bias, self.causal) for _ in
                                     range(self.num_blocks)])

        # Output convolutions
        self.out_conv = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv1d(in_channels=self.ch_skip, out_channels=self.ch_end, kernel_size=1, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv1d(in_channels=self.ch_end, out_channels=self.num_classes, kernel_size=1, bias=True)
        )

    def forward(self, x):
        # create channels
        residual = self.start_conv(x)

        # pass data through the blocks
        skip_outputs = []
        for block in self.blocks:
            residual, skip = block(residual)
            skip_outputs.append(skip)

        # sum skip connections
        summed_skip = torch.cat(skip_outputs).sum(dim=0)

        # carry out output transform
        out = self.out_conv(summed_skip)

        return out[..., -self.output_len:] # indexing is important, if starting from 0th index, even overfitting the network fails for me


from penn_dataset import PennTreeCharDataset

if __name__ == '__main__':
    ds = PennTreeCharDataset(32, 8, is_test=True)
    print("done")

    layer = WavenetLayer(dilation=1)
    block = WavenetBlock()
    model = WaveNet()
    batch = 16
    ch = 16
    len = 64
    input = torch.randn((batch, ch, len))
    # layer(input)
    # block(input)

    model(input)
    print("out")

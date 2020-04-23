import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from metrics import MetricsCalculator
from penn_dataset import PennTreeCharDataset, PennTreeSentenceDataset
from wavenet import WaveNet


class WaveNetModule(pl.LightningModule):

    def __init__(self, hparams, loss_fn=F.cross_entropy, log_grads: bool = False, use_sentence_split: bool = True):
        super().__init__()

        """Configuration flags"""
        self.use_sentence_split = use_sentence_split
        self.log_grads = log_grads

        """Dataset"""
        self.batch_size = hparams.batch_size
        self.output_length = hparams.out_len
        self.win_len = hparams.win_len
        self._setup_dataloaders()

        """Training"""
        self.loss_fn = loss_fn
        self.lr = hparams.lr

        """Embedding"""
        self.embedding_dim = hparams.emb_dim
        self.embedding = nn.Embedding(self.num_classes, self.embedding_dim)
        self.embedding.weight = nn.Parameter(torch.eye(self.embedding_dim), requires_grad=False)

        """Metrics"""
        self.metrics = MetricsCalculator(["accuracy", "precision", "recall", "f1"])

        """Model"""
        self.model = WaveNet(num_blocks=hparams.num_blocks, num_layers=hparams.num_layers, num_classes=self.num_classes,
                             output_len=self.output_length, ch_start=self.embedding_dim,
                             ch_residual=hparams.ch_residual, ch_dilation=hparams.ch_dilation, ch_skip=hparams.ch_skip,
                             ch_end=hparams.ch_end, kernel_size=hparams.kernel_size, bias=True)

    def forward(self, x):
        return self.model(x)

    def _forward_batch(self, batch):
        x, y = batch
        x_emb = self._embed(x)
        y_hat = self.forward(x_emb)
        return self.loss_fn(y_hat, y), y, torch.argmax(y_hat, dim=1)

    def _embed(self, x):
        x_emb = self.embedding(x).permute(0, 2, 1)
        return x_emb

    def training_step(self, batch, batch_idx):
        loss, true, preds = self._forward_batch(batch)

        return {"loss": loss, "log": (self.metrics.generate_logs(loss, preds, true, "train"))}

    def training_step_end(self, training_out):
        tensorboard_logs = self.metrics.generate_mean_metrics(training_out["log"], "train")
        return {"loss": training_out["loss"],
                "progress_bar": tensorboard_logs,
                "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, true, preds = self._forward_batch(batch)

        return self.metrics.generate_logs(loss, preds, true, "val")

    def validation_epoch_end(self, outputs):
        tensorboard_logs = self.metrics.generate_mean_metrics(outputs, "val")
        return {"progress_bar": tensorboard_logs,
                "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        loss, true, preds = self._forward_batch(batch)

        return self.metrics.generate_logs(loss, preds, true, "test")

    def test_epoch_end(self, outputs):
        tensorboard_logs = self.metrics.generate_mean_metrics(outputs, "test")
        return {"progress_bar": tensorboard_logs,
                "log": tensorboard_logs}

    def configure_optimizers(self):
        # can return multiple optimizers and learning_rate schedulers
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def _setup_dataloaders(self):

        ds = PennTreeSentenceDataset if self.use_sentence_split else PennTreeCharDataset

        self.dl_train = DataLoader(ds(self.win_len, self.output_length, is_train=True), self.batch_size, shuffle=True)
        self.dl_valid = DataLoader(ds(self.win_len, self.output_length, is_valid=True), self.batch_size)
        self.dl_test = DataLoader(ds(self.win_len, self.output_length, is_test=True), self.batch_size)

    @property
    def num_classes(self):
        return self.dl_train.dataset.num_chars

    def train_dataloader(self):
        return self.dl_train

    def val_dataloader(self):
        return self.dl_valid

    def test_dataloader(self):
        return self.dl_test

    def on_train_start(self):
        input = torch.ones((self.batch_size, self.embedding_dim, self.win_len))
        if torch.cuda.is_available():
            input = input.cuda()

        self.logger.experiment.add_graph(self.model, input)

    def on_after_backward(self):
        # example to inspect gradient information in tensorboard
        if self.log_grads and self.trainer.global_step % 100 == 0:  # don't make the tf file huge
            params = self.state_dict()
            for k, v in params.items():
                grads = v
                name = k
                self.logger.experiment.add_histogram(tag=name, values=grads,
                                                     global_step=self.trainer.global_step)


from args import get_args
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == "__main__":
    # argument parsing
    args = get_args()
    model = WaveNetModule(args, use_sentence_split=True)

    # most basic trainer, uses good defaults
    logger = TensorBoardLogger("lightning_logs", name="my_model")
    trainer = pl.Trainer(logger=logger, max_nb_epochs=1000, gpus=[0] if torch.cuda.is_available() else None,
                         overfit_pct=1, print_nan_grads=True, check_val_every_n_epoch=1)
    trainer.fit(model)

import os
import re
from glob import glob
from optparse import OptionParser

import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import UFData
from momentum_model import MomentumModel
from tqdm import tqdm

from network import SimpleNet


# TODO: Tensorboard
# TODO: Learning rate decay
# TODO: Tune temperature (~0.07?)
# TODO: Maybe sample from memory bank too?
# Done: Sample from encodings?


def get_args():
    parser = OptionParser()
    parser.add_option('--datadir', "--dd",
                      help='Directory contains 2D images.')
    parser.add_option("-g", '--gpu_id', dest="gpu_id", type='int',
                      help='GPU number, default is None (-g 0 means use gpu 0)')
    parser.add_option('--logdir', "--ld",
                      help='Directory for saving logs and checkpoints')
    parser.add_option('-f', '--features', default=128, type='int',
                      help='Dimension of the feature space.')
    parser.add_option('--learning-rate', '--lr', default=1e-4, type='float',
                      help='learning rate for the model')
    parser.add_option('--temperature', '--temp', default=1.00, type=float,
                      help='temperature parameter default: 1')
    parser.add_option('--momentum', default=0.999, type=float,
                      help='Momentum for target network.')
    parser.add_option('--batchsize', '--bs', dest='batchsize',
                      default=32, type='int', help='batch size for training')
    parser.add_option('-e', '--epochs', default=200, type='int',
                      help='Number of epochs to train')
    # parser.add_option('-m', '--model', dest='model',
    #                   default=False, help='load checkpoints')
    parser.add_option('--use_magnitude', action="store_true", default=False,
                      help='If specified, use image magnitude.')
    # parser.add_option('-x', '--sx', dest='sx',
    #                   default=256, type='int', help='image dim: x')
    # parser.add_option('-y', '--sy', dest='sy',
    #                   default=320, type='int', help='image dim: y')
    parser.add_option('--force_train_from_scratch', '--overwrite', action="store_true",
                      help="If specified, training will start from scratch."
                           " Otherwise, latest checkpoint (if any) will be used")
    parser.add_option('--fastmri', action="store_true", default=False,
                      help='If specified, use fastmri settings.')

    (options, args) = parser.parse_args()
    return options


class Trainer:

    def __init__(self):

        self.args = get_args()
        self.device = torch.device(f"cuda:{self.args.gpu_id}")
        print("Using device:", self.device)
        print("Using magnitude:", self.args.use_magnitude)

        self.checkpoint_directory = os.path.join(f"{self.args.logdir}", "checkpoints")
        os.makedirs(self.checkpoint_directory, exist_ok=True)

        self.dataset = UFData(self.args.datadir, magnitude=bool(self.args.use_magnitude), device=self.device,
                              fastmri=self.args.fastmri, complex=self.args.fastmri)
        self.dataloader = DataLoader(self.dataset, self.args.batchsize, shuffle=True, drop_last=True)

        self.model = MomentumModel(SimpleNet, momentum=self.args.momentum, temperature=self.args.temperature,
                                   device=self.device, magnitude=self.args.use_magnitude)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        self.start_epoch = 1
        if not self.args.force_train_from_scratch:
            self.restore_model()
        else:
            input("Training from scratch. Are you sure? (Ctrl+C to kill):")

    def restore_model(self):
        """Restore latest model checkpoint (if any) and continue training from there."""

        checkpoint_path = sorted(glob(os.path.join(self.checkpoint_directory, "*")),
                                 key=lambda x: int(re.match(".*[a-z]+(\d+).pth", x).group(1)))
        if checkpoint_path:
            checkpoint_path = checkpoint_path[-1]
            print(f"Found saved model at: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_dict"])
            self.start_epoch = checkpoint["epoch"] + 1  # Start at next epoch of saved model

            print(f"Finish restoring model. Resuming at epoch {self.start_epoch}")

        else:
            print("No saved model found. Training from scratch.")

    def save_model(self, epoch):
        """Save model checkpoint.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        """

        torch.save({
            "epoch": epoch,  # Epoch we just finished
            "state_dict": self.model.state_dict(),
            "optimizer_dict": self.optimizer.state_dict()
        }, os.path.join(self.checkpoint_directory, 'ckpt{}.pth'.format(epoch)))

    def train(self):
        """Train the model!"""

        losses = []

        for epoch in tqdm(range(self.start_epoch, self.args.epochs + 1), "Epoch"):

            self.model.train()
            for index, images in enumerate(tqdm(self.dataloader, "Step")):

                images = images.to(self.device)

                embeddings, target_embeddings = self.model(images)
                logits, labels = self.model.get_logits_labels(embeddings, target_embeddings)
                loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                embeddings.detach()
                target_embeddings.detach()

                with torch.no_grad():
                    self.model.update_target_network()
                    self.model.update_memory_bank(target_embeddings)

                if index % 20 == 0:
                    losses.append(loss.item())

            print(f"\n\n\tEpoch {epoch}. Loss {loss.item()}\n")
            np.save(os.path.join(self.args.logdir, "loss_all.npy"), np.array(losses))

            if epoch % 10 == 0:
                self.save_model(epoch)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    trainer = Trainer()
    trainer.train()

import numpy as np
import torch


class MomentumModel(torch.nn.Module):

    def __init__(self, network_class, feature_dim=128, max_buffer_size=65536, momentum=0.99975,
                 temperature=1.0, device=torch.device('cpu'), magnitude=False):
        super().__init__()
        print(f"Using parameters:\n   Momentum: {momentum}\n   Temperature: {temperature}")

        self.device = device
        self.feature_dim = feature_dim
        self.size = max_buffer_size
        self.temperature = temperature
        self.phase_augmentation = True
        self.magnitude = magnitude

        self.momentum = momentum
        self.network = network_class(num_classes=feature_dim).to(device)
        self.target_network = network_class(num_classes=feature_dim).to(device)
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(param.data)  # initialize with the same weights
            target_param.requires_grad = False  # not update by gradient

        self.register_buffer("memory_bank", torch.randn(feature_dim, max_buffer_size))  # Save memory bank with model
        self.memory_bank = torch.nn.functional.normalize(self.memory_bank.to(self.device), p=2, dim=0)
        self.register_buffer("current_index", torch.zeros(1, dtype=torch.int32))

    @torch.no_grad()
    def update_target_network(self):
        """Momentum update for delayed (target) network."""
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data = target_param.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def update_memory_bank(self, embeddings,
                           keep_rate=0.01695,
                           # keep_rate=0.0725
                           ):
        """Update memory bank queue with embeddings of new patches. Since many belong to the same original image, keep
        only a random subset of them to avoid high correlation between saved embeddings in the bank.

        Parameters
        ----------
        embeddings : Tensor
            Tensor with image encodings of shape (B, H, W, F).
        keep_rate : float
            The percentage of patch embeddings to store in the memory bank.
        """
        embeddings_flat = embeddings.contiguous().view((-1, 128))
        # Store at most 512 per step
        num_to_store = min(512, embeddings_flat.shape[0])  # int(embeddings_flat.shape[0] * keep_rate)
        indices = torch.randperm(embeddings_flat.shape[0])[:num_to_store]
        embeddings_flat = embeddings_flat[indices]

        assert self.size % num_to_store == 0  # For now, otherwise need handle overflow

        self.memory_bank[:, self.current_index: self.current_index + num_to_store] = embeddings_flat.T
        self.current_index = (self.current_index + num_to_store) % self.size

    def forward(self, images):

        net_images = self.prepare_images(images, phase_augmentation=self.phase_augmentation)
        embeddings = self.network(net_images)[0].permute(0, 2, 3, 1)  # B, H, W, F (channels last)
        with torch.no_grad():
            target_net_images = self.prepare_images(images, phase_augmentation=self.phase_augmentation)
            target_embeddings = self.target_network(target_net_images)[0].permute(0, 2, 3, 1).detach()

        return embeddings, target_embeddings

    def predict(self, images):
        with torch.no_grad():
            images = self.prepare_images(images, phase_augmentation=False)
            return self.target_network(images)[0].permute(0, 2, 3, 1).detach()

    def prepare_images(self, images, phase_augmentation=False):

        if phase_augmentation:
            images *= torch.exp(1j * 2 * np.pi * torch.rand(images.shape[0])).to(self.device)[:, None, None, None]

        if self.magnitude:
            images = torch.tensor(np.abs(images))
        else:
            images = torch.cat((images.real, images.imag), dim=1)

        return images

    def get_logits_labels(self, embeddings, target_embeddings, patch_sample_rate=0.05, bank_sample_rate=0.2):
        """Compute logits and labels used for computing the loss.

        Parameters
        ----------
        embeddings
        target_embeddings
        sample_rate

        Returns
        -------

        """
        embeddings_flat = embeddings.contiguous().view(-1, self.feature_dim)  # B*H*W, F
        target_embeddings_flat = target_embeddings.contiguous().view(-1, self.feature_dim)

        # Sample from embeddings! At least 1 per element in the batch
        num_to_sample = max(len(embeddings), int(len(embeddings_flat) * patch_sample_rate))
        indices = torch.randperm(len(embeddings_flat))[:num_to_sample]
        embeddings_flat = embeddings_flat[indices]
        target_embeddings_flat = target_embeddings_flat[indices]

        # Sample from Memory Bank!
        # indices = torch.randperm(len(self.memory_bank))[:int(len(self.memory_bank) * bank_sample_rate)]
        # memory_bank_sampled = self.memory_bank[:, indices]

        # Compare each patch against itself in "older" network. (N,1,F) x (N, F, 1) = (N, 1, 1) -> (N, 1)
        positive_logits = torch.bmm(embeddings_flat.view(-1, 1, self.feature_dim),
                                    target_embeddings_flat.view(-1, self.feature_dim, 1)).squeeze(dim=-1)

        # Compare each patch against the memory bank. (N, F) x (F, S) = (N, S).
        negative_logits = torch.mm(embeddings_flat, self.memory_bank)

        # Concat logits and create labels for 0th position
        logits = torch.cat([positive_logits, negative_logits], dim=1)  # (N, 1+S)
        logits /= self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device).detach()

        return logits, labels

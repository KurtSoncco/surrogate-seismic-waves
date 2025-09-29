from typing import Optional, Tuple

import torch
import torch.nn as nn


class DecoupledAutoencoder(nn.Module):
    """
    An implementation of the Decoupled Autoencoder (DAE) framework.

    This class manages a primary encoder (f) and decoder (g), along with an
    optional, weaker auxiliary decoder for Stage 1 training.
    References:
    - [1] "Complexity Matters: Rethinking the Latent Space for Generative Modeling"
      (https://proceedings.neurips.cc/paper_files/paper/2023/hash/5e8023f07625374c6fdf3aa08bb38e0e-Abstract-Conference.html)
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        aux_decoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        # If no separate aux_decoder is provided, the main decoder will be used
        # in Stage 1. This is for the case where the main decoder itself contains
        # regularization like Dropout to make it "weaker" during training[cite: 177].
        self.aux_decoder = aux_decoder if aux_decoder is not None else decoder

    def forward_stage1(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass for Stage 1 training.
        Uses the ENCODER and the AUXILIARY DECODER.
        """
        z = self.encoder(x)
        reconstruction = self.aux_decoder(z)
        return reconstruction, z

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard forward pass for Stage 2 training or inference.
        Uses the ENCODER and the MAIN DECODER.
        """
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes input data into the latent space."""
        with torch.no_grad():
            self.encoder.eval()
            z = self.encoder(x)
            self.encoder.train()
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes a latent vector back into the data space."""
        with torch.no_grad():
            self.decoder.eval()
            x_hat = self.decoder(z)
            self.decoder.train()
        return x_hat

    def save_encoder(self, path: str):
        """Saves the encoder (f) state dictionary to a file."""
        print(f"Saving encoder to {path}...")
        torch.save(self.encoder.state_dict(), path)

    def save_decoder(self, path: str):
        """Saves the main decoder (g) state dictionary to a file."""
        print(f"Saving decoder to {path}...")
        torch.save(self.decoder.state_dict(), path)

    @classmethod
    def load_encoder_from_file(cls, encoder_cls: type, path: str, **kwargs):
        """Loads a pre-trained encoder from a file to a new Encoder instance."""
        encoder = encoder_cls(**kwargs)
        encoder.load_state_dict(torch.load(path))
        return encoder

    @classmethod
    def load_decoder_from_file(cls, decoder_cls: type, path: str, **kwargs):
        """Loads a pre-trained decoder from a file to a new Decoder instance."""
        decoder = decoder_cls(**kwargs)
        decoder.load_state_dict(torch.load(path))
        return decoder

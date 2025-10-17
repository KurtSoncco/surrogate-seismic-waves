# model.py
"""Defines the FNO model architecture, including Encoder and OperatorDecoder."""

from typing import List

import torch
import torch.nn as nn


class FNO1D(nn.Module):
    """
    A 1D Fourier Neural Operator layer.

    The architecture is based on the paper:
    "Fourier Neural Operator for Parametric Partial Differential Equations"
    (https://arxiv.org/abs/2010.08895)

    Args:
        modes (int): Number of Fourier modes to use.
        width (int): Width of the input and output feature dimensions.

    Input Shape:
        (batch_size, sequence_length, width)

    Output Shape:
        (batch_size, sequence_length, width)
    """

    def __init__(self, modes, width):
        super(FNO1D, self).__init__()
        self.modes = modes
        self.width = width
        self.scale = 1 / (width**0.5)
        self.weights = nn.Parameter(
            self.scale * torch.randn(modes, width, width, 2, dtype=torch.float)
        )
        self.linear_transform = nn.Linear(width, width)
        self.activation = nn.ReLU()

    def compl_mul1d(self, input, weights):
        """
        Complex multiplication in the Fourier domain.
        Args:
            input: (batch, modes, width) - complex tensor
            weights: (modes, width, width) - complex tensor
        Returns:
            output: (batch, modes, width) - complex tensor
        """
        return torch.einsum("bmw,mwo->bmo", input, weights)

    def forward(self, x):
        """
        Forward pass of the FNO1D layer.
        Args:
            x: (batch, length, width) - input tensor
        Returns:
            (batch, length, width) - output tensor
        """
        batchsize, length, _ = x.shape
        
        # Ensure the input tensor is contiguous and has the right dtype for FFT
        x = x.contiguous()
        
        try:
            x_ft = torch.fft.rfft(x, dim=1, norm="ortho")
        except RuntimeError as e:
            if "MKL FFT error" in str(e):
                # Fallback: use a different FFT implementation or handle the error
                # Try with different normalization
                try:
                    x_ft = torch.fft.rfft(x, dim=1, norm="backward")
                except RuntimeError:
                    # If still failing, try without normalization
                    x_ft = torch.fft.rfft(x, dim=1)
            else:
                raise e
        
        out_ft = torch.zeros(
            batchsize, length // 2 + 1, self.width, dtype=torch.cfloat, device=x.device
        )

        modes_to_use = min(self.modes, length // 2)
        indices = torch.arange(0, modes_to_use, device=x.device)

        weights = torch.view_as_complex(self.weights)
        out_ft[:, indices, :] = self.compl_mul1d(
            x_ft[:, indices, :], weights[:modes_to_use, :, :]
        )

        try:
            x_fourier = torch.fft.irfft(out_ft, n=length, dim=1, norm="ortho")
        except RuntimeError as e:
            if "MKL FFT error" in str(e):
                # Fallback: use a different IFFT implementation
                try:
                    x_fourier = torch.fft.irfft(out_ft, n=length, dim=1, norm="backward")
                except RuntimeError:
                    # If still failing, try without normalization
                    x_fourier = torch.fft.irfft(out_ft, n=length, dim=1)
            else:
                raise e
        
        x_linear = self.linear_transform(x)

        return self.activation(x_fourier + x_linear)


class Encoder(nn.Module):
    """
    A configurable 1D CNN-based encoder to map input profiles to a latent space.

    The architecture is dynamically built based on the provided channel sizes,
    consisting of several Conv1D -> ReLU -> MaxPool1D blocks.

    Args:
        channels (List[int]): List of channel sizes for the convolutional layers.
                               The first element must be 1 (input channels).
        latent_dim (int): The dimension of the output latent space.
        kernel_size (int): The kernel size for all convolutional layers.
        pool_size (int): The kernel size for the MaxPool1D layers.

    Input Shape:
        (batch_size, input_channels, sequence_length)

    Output Shape:
        (batch_size, latent_dim)
    """

    def __init__(
        self,
        channels: List[int],
        latent_dim: int,
        kernel_size: int = 3,
        pool_size: int = 2,
    ):
        super().__init__()

        if not channels or channels[0] < 1:
            raise ValueError(
                "The 'channels' list must start with at least 1 input channel."
            )

        layers = []
        # Dynamically create the convolutional blocks
        for i in range(len(channels) - 1):
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels=channels[i],
                        out_channels=channels[i + 1],
                        kernel_size=kernel_size,
                        padding="same",
                    ),
                    nn.ReLU(),
                    nn.MaxPool1d(pool_size),
                ]
            )

        self.conv_layers = nn.Sequential(*layers)

        # Use a LazyLinear layer to automatically infer the input size
        self.fc = nn.LazyLinear(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input should be (B, C, L) where C is the number of input channels

        # Pass through convolutional blocks
        x = self.conv_layers(x)

        # Flatten all dimensions except the batch dimension
        x = x.flatten(1)

        # Final fully connected layer
        return self.fc(x)


class OperatorDecoder(nn.Module):
    """
    An FNO-based decoder that maps a latent representation to an output function.
    The architecture lifts the latent vector to a channel-rich sequence,
    processes it through a configurable stack of FNO layers, and projects it
    to the final output size.

    Args:
        latent_dim (int): The dimension of the input latent vector.
        output_size (int): The sequence length of the output function.
        fno_modes (int): The number of Fourier modes to use in each FNO layer.
        fno_width (int): The number of channels for the FNO layers.
        num_fno_layers (int): The number of FNO layers to stack.

    Input Shape:
        (batch_size, latent_dim)

    Output Shape:
        (batch_size, output_size)
    """

    def __init__(
        self,
        latent_dim: int,
        output_size: int,
        fno_modes: int,
        fno_width: int,
        num_fno_layers: int = 3,
    ):
        super().__init__()
        self.output_size = output_size
        self.fno_width = fno_width

        # First layer lifts the latent vector and reshapes it
        self.fc1 = nn.Linear(latent_dim, fno_width * output_size)

        # Dynamically create the stack of FNO layers
        fno_stack = [
            FNO1D(modes=fno_modes, width=fno_width) for _ in range(num_fno_layers)
        ]
        self.fno_stack = nn.Sequential(*fno_stack)

        # Final layer projects back to the desired output function (1 channel)
        self.fc2 = nn.Linear(fno_width, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Lift and reshape: (B, D_lat) -> (B, L * W) -> (B, L, W)
        x = self.fc1(z)
        x = x.view(z.shape[0], self.output_size, self.fno_width)

        # Process through FNO blocks: (B, L, W) -> (B, L, W)
        x = self.fno_stack(x)

        # Project to output function: (B, L, W) -> (B, L, 1) -> (B, L)
        x = self.fc2(x).squeeze(-1)
        return x


class EncoderOperatorModel(nn.Module):
    """
    A modular container that combines an encoder and a decoder.
    This design allows for flexible swapping of components.

    Args:
        encoder (nn.Module): An initialized encoder module.
        decoder (nn.Module): An initialized decoder module.

    Input Shape:
        Matches the input shape of the provided encoder.

    Output Shape:
        Matches the output shape of the provided decoder.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# pipeline.py
"""
Main pipeline class that combines encoder, FNO processor, and decoder.
This is the core module for the latent FNO experiment.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .decoders import create_decoder
from .encoders import create_encoder
from .fno_processor import create_fno_processor


class LatentFNOPipeline(nn.Module):
    """
    Main pipeline for latent FNO experiments.

    This pipeline consists of:
    1. An encoder that maps input data to latent space
    2. An FNO processor that operates in the latent space
    3. A decoder that maps from latent space to output data

    The hypothesis is that operating FNO in the latent space can be more effective
    than operating directly on the raw data.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        latent_dim: int,
        encoder_type: str = "mlp",
        decoder_type: str = "mlp",
        fno_processor_type: str = "simple",
        encoder_config: Optional[Dict[str, Any]] = None,
        decoder_config: Optional[Dict[str, Any]] = None,
        fno_processor_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim

        # Default configurations
        encoder_config = encoder_config or {}
        decoder_config = decoder_config or {}
        fno_processor_config = fno_processor_config or {}

        # Create encoder
        self.encoder = create_encoder(
            encoder_type=encoder_type,
            input_dim=input_dim,
            latent_dim=latent_dim,
            **encoder_config,
        )

        # Create FNO processor
        self.fno_processor = create_fno_processor(
            processor_type=fno_processor_type,
            latent_dim=latent_dim,
            **fno_processor_config,
        )

        # Create decoder
        self.decoder = create_decoder(
            decoder_type=decoder_type,
            latent_dim=latent_dim,
            output_dim=output_dim,
            **decoder_config,
        )

        # Store configuration for reference
        self.config = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "latent_dim": latent_dim,
            "encoder_type": encoder_type,
            "decoder_type": decoder_type,
            "fno_processor_type": fno_processor_type,
            "encoder_config": encoder_config,
            "decoder_config": decoder_config,
            "fno_processor_config": fno_processor_config,
        }

    def forward(
        self, x: torch.Tensor, condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the entire pipeline.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            condition: Optional condition tensor for conditional FNO processors

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Encode input to latent space
        latent = self.encoder(x)

        # Process in latent space using FNO
        if condition is not None and hasattr(self.fno_processor, "forward"):
            # Check if the processor supports conditional processing
            try:
                processed_latent = self.fno_processor(latent, condition)
            except TypeError:
                # Fallback for processors that don't support conditions
                processed_latent = self.fno_processor(latent)
        else:
            processed_latent = self.fno_processor(latent)

        # Decode from latent space to output
        output = self.decoder(processed_latent)

        return output

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output."""
        return self.decoder(latent)

    def process_latent(
        self, latent: torch.Tensor, condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Process latent representation using FNO."""
        if condition is not None and hasattr(self.fno_processor, "forward"):
            try:
                return self.fno_processor(latent, condition)
            except TypeError:
                return self.fno_processor(latent)
        else:
            return self.fno_processor(latent)

    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Get the latent representation of input data."""
        with torch.no_grad():
            return self.encoder(x)

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of the pipeline."""
        return self.config.copy()

    def save_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        epoch: int = 0,
        loss: float = 0.0,
        **kwargs,
    ):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.config,
            "epoch": epoch,
            "loss": loss,
            **kwargs,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(
        cls, path: str, map_location: str = "cpu"
    ) -> Tuple["LatentFNOPipeline", Dict[str, Any]]:
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)

        # Create model from config
        config = checkpoint["config"]
        model = cls(**config)

        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])

        return model, checkpoint


class AblationStudyPipeline(LatentFNOPipeline):
    """
    Pipeline for ablation studies to test different components.

    This allows easy testing of:
    - Encoder-only (no FNO, no decoder)
    - Encoder + Decoder (no FNO)
    - Encoder + FNO (no decoder)
    - Full pipeline
    """

    def __init__(
        self,
        *args,
        use_encoder: bool = True,
        use_fno: bool = True,
        use_decoder: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.use_encoder = use_encoder
        self.use_fno = use_fno
        self.use_decoder = use_decoder

    def forward(
        self, x: torch.Tensor, condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with optional components."""

        if self.use_encoder:
            latent = self.encoder(x)
        else:
            # If no encoder, treat input as latent (assuming input_dim == latent_dim)
            if x.size(-1) != self.latent_dim:
                raise ValueError(
                    f"Input dimension {x.size(-1)} must match latent dimension {self.latent_dim} when encoder is disabled"
                )
            latent = x

        if self.use_fno:
            if condition is not None:
                try:
                    processed_latent = self.fno_processor(latent, condition)
                except TypeError:
                    processed_latent = self.fno_processor(latent)
            else:
                processed_latent = self.fno_processor(latent)
        else:
            processed_latent = latent

        if self.use_decoder:
            output = self.decoder(processed_latent)
        else:
            # If no decoder, return processed latent (assuming latent_dim == output_dim)
            if processed_latent.size(-1) != self.output_dim:
                raise ValueError(
                    f"Latent dimension {processed_latent.size(-1)} must match output dimension {self.output_dim} when decoder is disabled"
                )
            output = processed_latent

        return output


class EnsembleLatentFNOPipeline(nn.Module):
    """
    Ensemble of multiple LatentFNOPipeline models for improved performance.
    """

    def __init__(self, pipelines: list, aggregation_method: str = "mean"):
        super().__init__()

        self.pipelines = nn.ModuleList(pipelines)
        self.aggregation_method = aggregation_method

        if aggregation_method not in ["mean", "weighted_mean", "median"]:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")

    def forward(
        self, x: torch.Tensor, condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through ensemble."""
        outputs = []

        for pipeline in self.pipelines:
            output = pipeline(x, condition)
            outputs.append(output)

        # Stack outputs: (num_models, batch_size, output_dim)
        outputs = torch.stack(outputs, dim=0)

        if self.aggregation_method == "mean":
            return outputs.mean(dim=0)
        elif self.aggregation_method == "median":
            return outputs.median(dim=0)[0]
        elif self.aggregation_method == "weighted_mean":
            # For now, use equal weights - can be extended to learnable weights
            weights = torch.ones(outputs.size(0), device=outputs.device) / outputs.size(
                0
            )
            weights = weights.view(-1, 1, 1)  # (num_models, 1, 1)
            result = (outputs * weights).sum(dim=0)
            return result

        # Fallback: should not happen because constructor validates aggregation_method,
        # but make sure this function always either returns a Tensor or raises.
        raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

    def add_pipeline(self, pipeline: LatentFNOPipeline):
        """Add a new pipeline to the ensemble."""
        self.pipelines.append(pipeline)


def create_pipeline(
    input_dim: int,
    output_dim: int,
    latent_dim: int,
    encoder_type: str = "mlp",
    decoder_type: str = "mlp",
    fno_processor_type: str = "simple",
    encoder_config: Optional[Dict[str, Any]] = None,
    decoder_config: Optional[Dict[str, Any]] = None,
    fno_processor_config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> LatentFNOPipeline:
    """
    Factory function to create a LatentFNOPipeline.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        latent_dim: Latent dimension
        encoder_type: Type of encoder to use
        decoder_type: Type of decoder to use
        fno_processor_type: Type of FNO processor to use
        encoder_config: Configuration for encoder
        decoder_config: Configuration for decoder
        fno_processor_config: Configuration for FNO processor
        **kwargs: Additional arguments

    Returns:
        LatentFNOPipeline instance
    """
    return LatentFNOPipeline(
        input_dim=input_dim,
        output_dim=output_dim,
        latent_dim=latent_dim,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        fno_processor_type=fno_processor_type,
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        fno_processor_config=fno_processor_config,
        **kwargs,
    )

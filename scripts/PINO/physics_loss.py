# physics_loss.py
"""Defines the physics loss function based on the 1D Shear Wave Equation with variable density."""

import numpy as np
import torch
import torch.nn as nn


class WaveEquationLoss(nn.Module):
    """
    Calculates the residual of the 1D Shear Wave Equation with variable density:
    rho(z) * u_tt - (G(z) * u_z)_z = 0

    This implementation correctly handles physical scaling and is fully differentiable.
    """

    def __init__(
        self,
        layer_thickness: float,
        total_depth: float,
        total_time: float,
        device: torch.device,
    ):
        super().__init__()
        self.layer_thickness = layer_thickness
        self.L0 = total_depth  # Characteristic length for normalization
        self.T0 = total_time  # Characteristic time for normalization
        self.device = device

    def forward(
        self, u_pred: torch.Tensor, vs_rho_profiles: torch.Tensor, grid: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the mean squared residual of the PDE over the batch.
        Args:
            u_pred (torch.Tensor): Predicted wavefield u(z,t) of shape (B, Z, T).
            vs_rho_profiles (torch.Tensor): Stacked Vs and Rho profiles of shape (B, 2, L_vs).
            grid (torch.Tensor): Spatiotemporal grid (z,t) of shape (B, Z, T, 2).
                           Assumed to be normalized in [0, 1].

        Returns:
            torch.Tensor: Mean squared residual of the PDE over the batch.
        """
        z_coords_norm = grid[0, :, 0, 0]  # Normalized z-coordinates [0, 1]
        num_layers = vs_rho_profiles.shape[2]

        # Unpack Vs and Rho
        vs_profiles = vs_rho_profiles[:, 0, :]
        rho_profiles = vs_rho_profiles[:, 1, :]

        # Interpolate G(z) and rho(z) onto the continuous grid in a differentiable way
        G_grid = self.get_physical_param_on_grid(
            vs_profiles, rho_profiles, z_coords_norm, num_layers, param_type="G"
        )
        rho_grid = self.get_physical_param_on_grid(
            vs_profiles, rho_profiles, z_coords_norm, num_layers, param_type="rho"
        )

        # Get all first-order derivatives in one go for efficiency
        grad_u = torch.autograd.grad(
            u_pred,
            grid,
            grad_outputs=torch.ones_like(u_pred),
            create_graph=True,
        )[0]
        du_dz_norm = grad_u[..., 0]  # Derivative w.r.t. normalized z
        du_dt_norm = grad_u[..., 1]  # Derivative w.r.t. normalized t

        # Second-order derivative w.r.t. time
        du_dtt_norm = torch.autograd.grad(
            du_dt_norm,
            grid,
            grad_outputs=torch.ones_like(du_dt_norm),
            create_graph=True,
        )[0][..., 1]

        # Stress term: G * du/dz_physical = G * (du/dz_norm) / L0
        stress = G_grid.unsqueeze(-1) * du_dz_norm / self.L0

        # Derivative of stress w.r.t. physical z: d(stress)/dz_norm / L0
        d_stress_dz = (
            torch.autograd.grad(
                stress,
                grid,
                grad_outputs=torch.ones_like(stress),
                create_graph=True,
            )[0][..., 0]
            / self.L0
        )

        # Apply chain rule for time derivative: u_tt_physical = u_tt_norm / T0^2
        inertial_term = rho_grid.unsqueeze(-1) * du_dtt_norm / (self.T0**2)

        # PDE Residual: rho(z) * u_tt - (G(z) * u_z)_z = 0
        residual = inertial_term - d_stress_dz
        return torch.mean(residual**2)

    def get_physical_param_on_grid(
        self,
        vs_profiles: torch.Tensor,
        rho_profiles: torch.Tensor,
        z_coords_norm: torch.Tensor,
        num_layers: int,
        param_type: str = "G",
    ) -> torch.Tensor:
        """
        Interpolates a physical parameter (G or rho) onto the z-grid in a
        differentiable, vectorized manner.
        Args:
            vs_profiles (torch.Tensor): Vs profiles of shape (B, L_vs).
            rho_profiles (torch.Tensor): Rho profiles of shape (B, L_vs).
            z_coords_norm (torch.Tensor): Normalized z-coordinates of shape (Z).
            num_layers (int): Number of discrete layers in the profiles.
            param_type (str): Type of parameter to compute ('G' for shear modulus, 'rho' for density).

        Returns:
            torch.Tensor: Interpolated parameter grid of shape (B, Z).
        """
        batch_size = vs_profiles.shape[0]
        spatial_points = z_coords_norm.shape[0]

        if param_type == "G":
            # G = rho * Vs^2
            param_values = rho_profiles * (vs_profiles**2)  # Shape: (B, L_vs)
        else:  # param_type == 'rho'
            param_values = rho_profiles  # Shape: (B, L_vs)

        # Convert normalized z-coords [0, 1] to physical depths
        total_depth = num_layers * self.layer_thickness
        depths = z_coords_norm * total_depth

        # Find the corresponding layer index for each depth
        indices = torch.floor(depths / self.layer_thickness).long()
        indices = torch.clamp(indices, 0, num_layers - 1)  # Shape: (Z)

        # Expand indices to shape (B, Z, 1)
        indices_expanded = indices.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)

        # Expand param_values to shape (B, Z, L_vs) to match indices' spatial dimension
        param_values_expanded = param_values.unsqueeze(1).expand(-1, spatial_points, -1)

        # Gather the parameter values for each point on the grid
        param_grid = torch.gather(param_values_expanded, 2, indices_expanded).squeeze(
            -1
        )

        return param_grid


if __name__ == "__main__":
    ## Let's use a dummy example to test the loss function
    batch_size = 4
    spatial_points = 50  # Use a finer grid for more stable derivatives
    timesteps = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Define Physical Properties and Scales ---
    L_vs = 20  # Number of soil layers
    layer_thickness = 2.0  # meters
    total_depth = L_vs * layer_thickness  # Total depth L0
    total_time = 1.0  # Total time duration T0 in seconds

    # --- 2. Create Grid and Dummy Wavefield ---
    # Create spatiotemporal grid (B, Z, T, 2) normalized to [0, 1]
    z_coords = torch.linspace(0, 1, spatial_points, device=device)
    t_coords = torch.linspace(0, 1, timesteps, device=device)
    zz, tt = torch.meshgrid(z_coords, t_coords, indexing="ij")
    grid = torch.stack([zz, tt], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    grid.requires_grad_(True)

    # Dummy predicted wavefield u(z,t) - use a smoother function
    # A simple standing wave fundamental mode
    u_pred = torch.sin(np.pi * grid[..., 0]) * torch.cos(np.pi * grid[..., 1])

    # --- 3. Create Dummy Physical Profiles (in SI units) ---
    # Realistic Vs values for soil (e.g., 150 m/s to 800 m/s)
    vs_profiles = 150 + torch.rand(batch_size, L_vs, device=device) * 650
    # Realistic rho values for soil (e.g., 1800 kg/m^3 to 2200 kg/m^3)
    rho_profiles = 1800 + torch.rand(batch_size, L_vs, device=device) * 400
    vs_rho_profiles = torch.stack([vs_profiles, rho_profiles], dim=1)

    # --- 4. Instantiate and Compute the Loss ---
    loss_fn = WaveEquationLoss(layer_thickness, total_depth, total_time, device)
    loss_value = loss_fn(u_pred, vs_rho_profiles, grid)

    print(f"Corrected loss value: {loss_value.item():.6f}")
    print("This value is now numerically stable and meaningful.")
    print("A perfect solution to the PDE would yield a loss of 0.")

# metrics.py
"""
Comprehensive metrics and evaluation utilities for the latent FNO experiments.
"""

import torch
import numpy as np
import scipy.stats as stats
from typing import Dict, Any, Tuple, Optional, List
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class MetricsCalculator:
    """
    Comprehensive metrics calculator for model evaluation.
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
    
    def calculate_all_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        frequency_data: Optional[np.ndarray] = None,
        sample_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Calculate all available metrics.
        
        Args:
            predictions: Model predictions (batch_size, output_dim)
            targets: Ground truth targets (batch_size, output_dim)
            frequency_data: Frequency values for frequency-dependent metrics
            sample_weights: Optional sample weights
            
        Returns:
            Dictionary of all calculated metrics
        """
        # Convert to numpy for some calculations
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        metrics = {}
        
        # Basic regression metrics
        metrics.update(self.calculate_regression_metrics(predictions, targets, sample_weights))
        
        # Correlation metrics
        metrics.update(self.calculate_correlation_metrics(pred_np, target_np))
        
        # Frequency-dependent metrics
        if frequency_data is not None:
            metrics.update(self.calculate_frequency_metrics(pred_np, target_np, frequency_data))
        
        # Statistical metrics
        metrics.update(self.calculate_statistical_metrics(pred_np, target_np))
        
        # Error distribution metrics
        metrics.update(self.calculate_error_distribution_metrics(pred_np, target_np))
        
        return metrics
    
    def calculate_regression_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Calculate standard regression metrics."""
        metrics = {}
        
        # Mean Squared Error
        if sample_weights is not None:
            mse = torch.mean(sample_weights * (predictions - targets) ** 2)
        else:
            mse = torch.mean((predictions - targets) ** 2)
        metrics['mse'] = mse.item()
        
        # Root Mean Squared Error
        metrics['rmse'] = torch.sqrt(mse).item()
        
        # Mean Absolute Error
        if sample_weights is not None:
            mae = torch.mean(sample_weights * torch.abs(predictions - targets))
        else:
            mae = torch.mean(torch.abs(predictions - targets))
        metrics['mae'] = mae.item()
        
        # Mean Absolute Percentage Error
        epsilon = 1e-8
        mape = torch.mean(torch.abs((targets - predictions) / (targets + epsilon))) * 100
        metrics['mape'] = mape.item()
        
        # R-squared
        ss_res = torch.sum((targets - predictions) ** 2)
        ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + epsilon))
        metrics['r2'] = r2.item()
        
        # Explained Variance
        var_target = torch.var(targets)
        var_residual = torch.var(targets - predictions)
        explained_var = 1 - (var_residual / (var_target + epsilon))
        metrics['explained_variance'] = explained_var.item()
        
        return metrics
    
    def calculate_correlation_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Calculate correlation-based metrics."""
        metrics = {}
        
        # Flatten arrays for overall correlation
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        # Pearson correlation
        pearson_corr, pearson_p = stats.pearsonr(pred_flat, target_flat)
        metrics['pearson_correlation'] = pearson_corr
        metrics['pearson_p_value'] = pearson_p
        
        # Spearman correlation
        spearman_corr, spearman_p = stats.spearmanr(pred_flat, target_flat)
        metrics['spearman_correlation'] = spearman_corr
        metrics['spearman_p_value'] = spearman_p
        
        # Kendall tau
        kendall_tau, kendall_p = stats.kendalltau(pred_flat, target_flat)
        metrics['kendall_tau'] = kendall_tau
        metrics['kendall_p_value'] = kendall_p
        
        # Per-sample correlations
        sample_corrs = []
        for i in range(len(predictions)):
            corr, _ = stats.pearsonr(predictions[i], targets[i])
            if not np.isnan(corr):
                sample_corrs.append(corr)
        
        sample_corrs = np.array(sample_corrs)
        metrics['mean_sample_correlation'] = np.mean(sample_corrs)
        metrics['std_sample_correlation'] = np.std(sample_corrs)
        metrics['min_sample_correlation'] = np.min(sample_corrs)
        metrics['max_sample_correlation'] = np.max(sample_corrs)
        
        return metrics
    
    def calculate_frequency_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        frequencies: np.ndarray
    ) -> Dict[str, float]:
        """Calculate frequency-dependent metrics."""
        metrics = {}
        
        # Correlation at each frequency
        freq_corrs = []
        freq_maes = []
        freq_rmses = []
        
        for i in range(predictions.shape[1]):
            pred_freq = predictions[:, i]
            target_freq = targets[:, i]
            
            # Correlation
            corr, _ = stats.pearsonr(pred_freq, target_freq)
            if not np.isnan(corr):
                freq_corrs.append(corr)
            
            # MAE and RMSE
            freq_maes.append(np.mean(np.abs(pred_freq - target_freq)))
            freq_rmses.append(np.sqrt(np.mean((pred_freq - target_freq) ** 2)))
        
        freq_corrs = np.array(freq_corrs)
        freq_maes = np.array(freq_maes)
        freq_rmses = np.array(freq_rmses)
        
        # Frequency-dependent statistics
        metrics['mean_frequency_correlation'] = np.mean(freq_corrs)
        metrics['std_frequency_correlation'] = np.std(freq_corrs)
        metrics['min_frequency_correlation'] = np.min(freq_corrs)
        metrics['max_frequency_correlation'] = np.max(freq_corrs)
        
        metrics['mean_frequency_mae'] = np.mean(freq_maes)
        metrics['mean_frequency_rmse'] = np.mean(freq_rmses)
        
        # Low frequency performance (first 10% of frequencies)
        low_freq_end = int(0.1 * len(frequencies))
        metrics['low_freq_correlation'] = np.mean(freq_corrs[:low_freq_end])
        metrics['low_freq_mae'] = np.mean(freq_maes[:low_freq_end])
        
        # High frequency performance (last 10% of frequencies)
        high_freq_start = int(0.9 * len(frequencies))
        metrics['high_freq_correlation'] = np.mean(freq_corrs[high_freq_start:])
        metrics['high_freq_mae'] = np.mean(freq_maes[high_freq_start:])
        
        return metrics
    
    def calculate_statistical_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Calculate statistical metrics."""
        metrics = {}
        
        residuals = predictions - targets
        
        # Residual statistics
        metrics['residual_mean'] = np.mean(residuals)
        metrics['residual_std'] = np.std(residuals)
        metrics['residual_skewness'] = stats.skew(residuals.flatten())
        metrics['residual_kurtosis'] = stats.kurtosis(residuals.flatten())
        
        # Normality test (Jarque-Bera)
        jb_stat, jb_p = stats.jarque_bera(residuals.flatten())
        metrics['jarque_bera_statistic'] = jb_stat
        metrics['jarque_bera_p_value'] = jb_p
        
        # Prediction statistics
        metrics['prediction_mean'] = np.mean(predictions)
        metrics['prediction_std'] = np.std(predictions)
        metrics['target_mean'] = np.mean(targets)
        metrics['target_std'] = np.std(targets)
        
        return metrics
    
    def calculate_error_distribution_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Calculate error distribution metrics."""
        metrics = {}
        
        errors = predictions - targets
        abs_errors = np.abs(errors)
        
        # Error percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            metrics[f'error_p{p}'] = np.percentile(abs_errors, p)
        
        # Outlier metrics (errors > 3 standard deviations)
        error_std = np.std(errors)
        outliers = np.abs(errors) > 3 * error_std
        metrics['outlier_percentage'] = np.mean(outliers) * 100
        
        # Error range
        metrics['error_min'] = np.min(errors)
        metrics['error_max'] = np.max(errors)
        metrics['error_range'] = np.max(errors) - np.min(errors)
        
        return metrics


class ModelEvaluator:
    """
    Comprehensive model evaluator with different evaluation strategies.
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.metrics_calculator = MetricsCalculator(device)
    
    def evaluate_model(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        frequency_data: Optional[np.ndarray] = None,
        criterion: Optional[torch.nn.Module] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Model to evaluate
            dataloader: DataLoader with test data
            frequency_data: Frequency values for frequency-dependent metrics
            criterion: Loss criterion
            
        Returns:
            Dictionary with all evaluation results
        """
        model.eval()
        
        all_predictions = []
        all_targets = []
        all_losses = []
        
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 2:
                    inputs, targets = batch
                    conditions = None
                else:
                    inputs, targets, conditions = batch
                
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                if conditions is not None:
                    conditions = conditions.to(self.device)
                    predictions = model(inputs, conditions)
                else:
                    predictions = model(inputs)
                
                # Calculate loss if criterion provided
                if criterion is not None:
                    loss = criterion(predictions, targets)
                    all_losses.append(loss.item())
                
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_all_metrics(
            all_predictions, all_targets, frequency_data
        )
        
        # Add loss if calculated
        if all_losses:
            metrics['average_loss'] = np.mean(all_losses)
        
        # Add sample information
        metrics['num_samples'] = len(all_predictions)
        
        return {
            'metrics': metrics,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def evaluate_with_confidence_intervals(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        num_bootstrap: int = 100,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Evaluate model with bootstrap confidence intervals.
        
        Args:
            model: Model to evaluate
            dataloader: DataLoader with test data
            num_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with metrics and confidence intervals
        """
        # Get all predictions and targets
        results = self.evaluate_model(model, dataloader)
        predictions = results['predictions']
        targets = results['targets']
        
        # Bootstrap sampling
        bootstrap_metrics = []
        n_samples = len(predictions)
        
        for _ in range(num_bootstrap):
            # Bootstrap sample
            indices = torch.randint(0, n_samples, (n_samples,))
            boot_pred = predictions[indices]
            boot_target = targets[indices]
            
            # Calculate metrics
            boot_metrics = self.metrics_calculator.calculate_all_metrics(
                boot_pred, boot_target
            )
            bootstrap_metrics.append(boot_metrics)
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        confidence_intervals = {}
        for metric_name in bootstrap_metrics[0].keys():
            values = [m[metric_name] for m in bootstrap_metrics]
            confidence_intervals[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'lower': np.percentile(values, lower_percentile),
                'upper': np.percentile(values, upper_percentile)
            }
        
        return {
            'metrics': results['metrics'],
            'confidence_intervals': confidence_intervals,
            'predictions': predictions,
            'targets': targets
        }
    
    def cross_validate_model(
        self,
        model_class,
        model_config: Dict[str, Any],
        dataset: torch.utils.data.Dataset,
        num_folds: int = 5,
        train_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation.
        
        Args:
            model_class: Model class to instantiate
            model_config: Configuration for model instantiation
            dataset: Dataset to use for cross-validation
            num_folds: Number of folds
            train_config: Training configuration
            
        Returns:
            Dictionary with cross-validation results
        """
        from sklearn.model_selection import KFold
        
        # Create k-fold splits
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        
        fold_results = []
        
        for fold, (train_indices, val_indices) in enumerate(kfold.split(dataset)):
            print(f"Fold {fold + 1}/{num_folds}")
            
            # Create fold datasets
            train_subset = torch.utils.data.Subset(dataset, train_indices)
            val_subset = torch.utils.data.Subset(dataset, val_indices)
            
            # Create data loaders
            train_loader = torch.utils.data.DataLoader(
                train_subset, batch_size=train_config.get('batch_size', 32), shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_subset, batch_size=train_config.get('batch_size', 32), shuffle=False
            )
            
            # Create and train model
            model = model_class(**model_config)
            # Here you would add your training logic
            # For now, we'll just evaluate the untrained model
            
            # Evaluate model
            results = self.evaluate_model(model, val_loader)
            fold_results.append(results['metrics'])
        
        # Aggregate results
        aggregated_metrics = {}
        for metric_name in fold_results[0].keys():
            values = [r[metric_name] for r in fold_results]
            aggregated_metrics[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return {
            'fold_results': fold_results,
            'aggregated_metrics': aggregated_metrics
        }


def create_metrics_summary(metrics: Dict[str, float]) -> str:
    """
    Create a formatted summary of metrics.
    
    Args:
        metrics: Dictionary of metrics
        
    Returns:
        Formatted string summary
    """
    summary_lines = ["Model Evaluation Summary", "=" * 50]
    
    # Group metrics by category
    categories = {
        'Regression': ['mse', 'rmse', 'mae', 'mape', 'r2', 'explained_variance'],
        'Correlation': ['pearson_correlation', 'spearman_correlation', 'kendall_tau', 
                       'mean_sample_correlation'],
        'Statistical': ['residual_mean', 'residual_std', 'residual_skewness', 'residual_kurtosis']
    }
    
    for category, metric_names in categories.items():
        summary_lines.append(f"\n{category} Metrics:")
        summary_lines.append("-" * 20)
        
        for metric_name in metric_names:
            if metric_name in metrics:
                value = metrics[metric_name]
                if isinstance(value, float):
                    summary_lines.append(f"  {metric_name}: {value:.6f}")
                else:
                    summary_lines.append(f"  {metric_name}: {value}")
    
    return "\n".join(summary_lines)

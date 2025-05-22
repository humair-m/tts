"""
Enhanced Audio Processing Utilities
Provides optimized functions for audio processing, alignment, and visualization.
"""

from monotonic_align import maximum_path
from monotonic_align import mask_from_lens
from monotonic_align.core import maximum_path_c
import numpy as np
import torch
import copy
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import matplotlib.pyplot as plt
from munch import Munch
import logging
import os
from pathlib import Path
from typing import Optional, Tuple, List, Union, Dict, Any
import warnings


def maximum_path(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Cython optimized version for finding maximum alignment path.
    
    Args:
        neg_cent: Negativecentroid tensor of shape [batch, target_time, source_time]
        mask: Mask tensor of shape [batch, target_time, source_time]
    
    Returns:
        torch.Tensor: Optimal alignment path
        
    Raises:
        ValueError: If input tensors have incompatible shapes
        RuntimeError: If Cython optimization fails
    """
    if neg_cent.shape != mask.shape:
        raise ValueError(f"Shape mismatch: neg_cent {neg_cent.shape} != mask {mask.shape}")
    
    try:
        device = neg_cent.device
        dtype = neg_cent.dtype
        
        # Convert to numpy with proper memory layout
        neg_cent_np = np.ascontiguousarray(
            neg_cent.detach().cpu().numpy().astype(np.float32)
        )
        path = np.ascontiguousarray(
            np.zeros(neg_cent.shape, dtype=np.int32)
        )
        
        # Calculate maximum lengths along each dimension
        t_t_max = np.ascontiguousarray(
            mask.sum(1)[:, 0].detach().cpu().numpy().astype(np.int32)
        )
        t_s_max = np.ascontiguousarray(
            mask.sum(2)[:, 0].detach().cpu().numpy().astype(np.int32)
        )
        
        # Call Cython optimized function
        maximum_path_c(path, neg_cent_np, t_t_max, t_s_max)
        
        return torch.from_numpy(path).to(device=device, dtype=dtype)
        
    except Exception as e:
        raise RuntimeError(f"Maximum path computation failed: {str(e)}")


def get_data_path_list(train_path: Optional[str] = None, 
                      val_path: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """
    Load training and validation data paths from text files.
    
    Args:
        train_path: Path to training data list file
        val_path: Path to validation data list file
    
    Returns:
        Tuple of training and validation file lists
        
    Raises:
        FileNotFoundError: If specified files don't exist
        IOError: If files cannot be read
    """
    if train_path is None:
        train_path = "Data/train_list.txt"
    if val_path is None:
        val_path = "Data/val_list.txt"
    
    train_list, val_list = [], []
    
    # Load training data paths
    try:
        train_path_obj = Path(train_path)
        if not train_path_obj.exists():
            raise FileNotFoundError(f"Training data file not found: {train_path}")
            
        with open(train_path, 'r', encoding='utf-8', errors='ignore') as f:
            train_list = [line.strip() for line in f.readlines() if line.strip()]
            
    except IOError as e:
        raise IOError(f"Error reading training data file {train_path}: {str(e)}")
    
    # Load validation data paths
    try:
        val_path_obj = Path(val_path)
        if not val_path_obj.exists():
            raise FileNotFoundError(f"Validation data file not found: {val_path}")
            
        with open(val_path, 'r', encoding='utf-8', errors='ignore') as f:
            val_list = [line.strip() for line in f.readlines() if line.strip()]
            
    except IOError as e:
        raise IOError(f"Error reading validation data file {val_path}: {str(e)}")
    
    # Validate that lists are not empty
    if not train_list:
        warnings.warn(f"Training data list is empty: {train_path}")
    if not val_list:
        warnings.warn(f"Validation data list is empty: {val_path}")
    
    return train_list, val_list


def length_to_mask(lengths: torch.Tensor) -> torch.Tensor:
    """
    Convert sequence lengths to boolean mask tensor.
    
    Args:
        lengths: Tensor of sequence lengths [batch_size]
    
    Returns:
        torch.Tensor: Boolean mask tensor [batch_size, max_length]
        
    Raises:
        ValueError: If lengths tensor is empty or has invalid values
    """
    if lengths.numel() == 0:
        raise ValueError("Lengths tensor cannot be empty")
    
    if torch.any(lengths < 0):
        raise ValueError("All lengths must be non-negative")
    
    max_len = lengths.max().item()
    if max_len == 0:
        return torch.zeros(lengths.shape[0], 1, dtype=torch.bool, device=lengths.device)
    
    # Create mask efficiently
    mask = torch.arange(max_len, device=lengths.device, dtype=lengths.dtype)
    mask = mask.unsqueeze(0).expand(lengths.shape[0], -1)
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    
    return mask


def log_norm(x: torch.Tensor, 
            mean: float = -4, 
            std: float = 4, 
            dim: int = 2, 
            eps: float = 1e-8) -> torch.Tensor:
    """
    Apply log normalization: normalized log mel -> mel -> norm -> log(norm).
    
    Args:
        x: Input tensor
        mean: Mean value for denormalization
        std: Standard deviation for denormalization
        dim: Dimension along which to compute norm
        eps: Small value to prevent log(0)
    
    Returns:
        torch.Tensor: Log-normalized tensor
        
    Raises:
        ValueError: If input tensor is empty or has invalid dimensions
    """
    if x.numel() == 0:
        raise ValueError("Input tensor cannot be empty")
    
    if dim >= x.ndim or dim < -x.ndim:
        raise ValueError(f"Invalid dimension {dim} for tensor with {x.ndim} dimensions")
    
    try:
        # Denormalize from log mel to mel
        mel = torch.exp(x * std + mean)
        
        # Compute norm along specified dimension
        norm_val = mel.norm(dim=dim)
        
        # Add epsilon to prevent log(0) and take log
        log_norm_val = torch.log(norm_val + eps)
        
        return log_norm_val
        
    except Exception as e:
        raise RuntimeError(f"Log normalization failed: {str(e)}")


def get_image(arrs: Union[np.ndarray, torch.Tensor], 
             title: str = "", 
             colormap: str = 'viridis',
             figsize: Tuple[int, int] = (8, 6),
             save_path: Optional[str] = None) -> plt.Figure:
    """
    Create matplotlib figure from array data with enhanced visualization options.
    
    Args:
        arrs: Input array or tensor to visualize
        title: Title for the plot
        colormap: Colormap to use for visualization
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure
    
    Returns:
        matplotlib.figure.Figure: The created figure
        
    Raises:
        ValueError: If input array is empty or has invalid shape
        IOError: If save path is invalid
    """
    # Set non-interactive backend
    plt.switch_backend('agg')
    
    # Convert tensor to numpy if needed
    if isinstance(arrs, torch.Tensor):
        arrs = arrs.detach().cpu().numpy()
    
    if arrs.size == 0:
        raise ValueError("Input array cannot be empty")
    
    # Create figure with enhanced settings
    fig, ax = plt.subplots(figsize=figsize)
    
    try:
        # Display image with colormap
        im = ax.imshow(arrs, cmap=colormap, aspect='auto')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Set title if provided
        if title:
            ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Improve layout
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            save_path_obj = Path(save_path)
            save_path_obj.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
        
    except Exception as e:
        plt.close(fig)
        raise RuntimeError(f"Image creation failed: {str(e)}")


def recursive_munch(d: Any) -> Any:
    """
    Recursively convert nested dictionaries and lists to Munch objects.
    
    Args:
        d: Input data structure (dict, list, or other)
    
    Returns:
        Converted data structure with Munch objects
        
    Raises:
        TypeError: If input contains unsupported types
        RecursionError: If input has circular references
    """
    try:
        if isinstance(d, dict):
            return Munch((k, recursive_munch(v)) for k, v in d.items())
        elif isinstance(d, list):
            return [recursive_munch(v) for v in d]
        elif isinstance(d, tuple):
            return tuple(recursive_munch(v) for v in d)
        else:
            return d
            
    except RecursionError:
        raise RecursionError("Circular reference detected in data structure")
    except Exception as e:
        raise TypeError(f"Unsupported type conversion: {str(e)}")


def log_print(message: str, 
             logger: Optional[logging.Logger] = None,
             level: str = 'info',
             print_to_console: bool = True) -> None:
    """
    Enhanced logging function with multiple output options.
    
    Args:
        message: Message to log
        logger: Logger instance (optional)
        level: Logging level ('debug', 'info', 'warning', 'error', 'critical')
        print_to_console: Whether to also print to console
        
    Raises:
        ValueError: If invalid logging level is provided
    """
    # Validate logging level
    valid_levels = ['debug', 'info', 'warning', 'error', 'critical']
    if level.lower() not in valid_levels:
        raise ValueError(f"Invalid logging level: {level}. Must be one of {valid_levels}")
    
    # Log to logger if provided
    if logger is not None:
        log_method = getattr(logger, level.lower())
        log_method(message)
    
    # Print to console if requested
    if print_to_console:
        # Add timestamp for console output
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        console_message = f"[{timestamp}] {level.upper()}: {message}"
        print(console_message)


# Additional utility functions for enhanced functionality

def validate_audio_tensor(audio: torch.Tensor, 
                         sample_rate: int = 22050,
                         min_length: int = 1000) -> bool:
    """
    Validate audio tensor properties.
    
    Args:
        audio: Audio tensor to validate
        sample_rate: Expected sample rate
        min_length: Minimum required length
    
    Returns:
        bool: True if valid, raises exception otherwise
    """
    if not isinstance(audio, torch.Tensor):
        raise TypeError("Audio must be a torch.Tensor")
    
    if audio.ndim not in [1, 2]:
        raise ValueError(f"Audio must be 1D or 2D, got {audio.ndim}D")
    
    if audio.shape[-1] < min_length:
        raise ValueError(f"Audio too short: {audio.shape[-1]} < {min_length}")
    
    if torch.any(torch.isnan(audio)) or torch.any(torch.isinf(audio)):
        raise ValueError("Audio contains NaN or Inf values")
    
    return True


def compute_spectral_features(audio: torch.Tensor,
                            sample_rate: int = 22050,
                            n_fft: int = 1024,
                            hop_length: int = 256,
                            n_mels: int = 80) -> Dict[str, torch.Tensor]:
    """
    Compute various spectral features from audio.
    
    Args:
        audio: Audio tensor [batch_size, time] or [time]
        sample_rate: Sample rate of audio
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_mels: Number of mel bins
    
    Returns:
        Dictionary containing spectral features
    """
    validate_audio_tensor(audio)
    
    # Ensure audio is 2D
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    
    features = {}
    
    # Mel spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    features['mel_spectrogram'] = mel_transform(audio)
    features['log_mel_spectrogram'] = torch.log(features['mel_spectrogram'] + 1e-8)
    
    # MFCC
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=13,
        melkwargs={
            'n_fft': n_fft,
            'hop_length': hop_length,
            'n_mels': n_mels
        }
    )
    
    features['mfcc'] = mfcc_transform(audio)
    
    return features


class EnhancedAudioProcessor:
    """
    Enhanced audio processing class with comprehensive functionality.
    """
    
    def __init__(self, 
                 sample_rate: int = 22050,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 n_mels: int = 80):
        """
        Initialize the audio processor.
        
        Args:
            sample_rate: Target sample rate
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel bins
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # Initialize transforms
        self._setup_transforms()
    
    def _setup_transforms(self):
        """Setup audio transforms."""
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=13,
            melkwargs={
                'n_fft': self.n_fft,
                'hop_length': self.hop_length,
                'n_mels': self.n_mels
            }
        )
    
    def process_audio(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process audio and extract features.
        
        Args:
            audio: Input audio tensor
        
        Returns:
            Dictionary of extracted features
        """
        validate_audio_tensor(audio)
        
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        
        features = {}
        features['mel_spectrogram'] = self.mel_transform(audio)
        features['log_mel_spectrogram'] = torch.log(features['mel_spectrogram'] + 1e-8)
        features['mfcc'] = self.mfcc_transform(audio)
        
        return features
    
    def visualize_features(self, 
                          features: Dict[str, torch.Tensor],
                          save_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
        """
        Visualize extracted features.
        
        Args:
            features: Dictionary of features
            save_dir: Optional directory to save figures
        
        Returns:
            Dictionary of matplotlib figures
        """
        figures = {}
        
        for name, feature in features.items():
            if feature.ndim >= 2:
                # Take first batch item if batch dimension exists
                feat_2d = feature[0] if feature.ndim == 3 else feature
                
                save_path = None
                if save_dir:
                    save_path = os.path.join(save_dir, f"{name}.png")
                
                figures[name] = get_image(
                    feat_2d,
                    title=name.replace('_', ' ').title(),
                    save_path=save_path
                )
        
        return figures

# coding: utf-8
import os
import sys
import os.path as osp
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import warnings
from functools import reduce

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import _LRScheduler, OneCycleLR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiOptimizer:
    """
    Enhanced MultiOptimizer class for managing multiple optimizers and schedulers.
    
    This class allows you to manage multiple optimizers simultaneously, which is useful
    for complex models with different parameter groups requiring different optimization
    strategies (e.g., GANs, multi-task learning).
    
    Args:
        optimizers (Dict[str, Optimizer]): Dictionary of optimizers keyed by name
        schedulers (Dict[str, _LRScheduler]): Dictionary of schedulers keyed by name
        
    Example:
        >>> optimizers = {'encoder': AdamW(encoder_params), 'decoder': AdamW(decoder_params)}
        >>> schedulers = {'encoder': scheduler1, 'decoder': scheduler2}
        >>> multi_optim = MultiOptimizer(optimizers, schedulers)
    """
    
    def __init__(self, optimizers: Dict[str, Optimizer] = None, 
                 schedulers: Dict[str, _LRScheduler] = None):
        self.optimizers = optimizers or {}
        self.schedulers = schedulers or {}
        self.keys = list(self.optimizers.keys())
        
        # Validate that all optimizer keys have corresponding schedulers if schedulers are provided
        if self.schedulers:
            missing_schedulers = set(self.keys) - set(self.schedulers.keys())
            if missing_schedulers:
                warnings.warn(f"Missing schedulers for optimizers: {missing_schedulers}")
        
        # Combine all parameter groups for unified access
        try:
            self.param_groups = reduce(
                lambda x, y: x + y, 
                [optimizer.param_groups for optimizer in self.optimizers.values()],
                []
            )
        except Exception as e:
            logger.warning(f"Failed to combine parameter groups: {e}")
            self.param_groups = []
    
    def add_optimizer(self, key: str, optimizer: Optimizer, 
                     scheduler: Optional[_LRScheduler] = None) -> None:
        """Add a new optimizer (and optionally scheduler) to the collection."""
        if key in self.optimizers:
            warnings.warn(f"Optimizer '{key}' already exists. Overwriting.")
        
        self.optimizers[key] = optimizer
        if scheduler is not None:
            self.schedulers[key] = scheduler
        
        self.keys = list(self.optimizers.keys())
        self._update_param_groups()
    
    def remove_optimizer(self, key: str) -> None:
        """Remove an optimizer and its scheduler from the collection."""
        if key not in self.optimizers:
            raise KeyError(f"Optimizer '{key}' not found")
        
        del self.optimizers[key]
        if key in self.schedulers:
            del self.schedulers[key]
        
        self.keys = list(self.optimizers.keys())
        self._update_param_groups()
    
    def _update_param_groups(self) -> None:
        """Update the combined parameter groups."""
        try:
            self.param_groups = reduce(
                lambda x, y: x + y,
                [optimizer.param_groups for optimizer in self.optimizers.values()],
                []
            )
        except Exception as e:
            logger.warning(f"Failed to update parameter groups: {e}")
            self.param_groups = []
    
    def state_dict(self) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Return the state dictionary of all optimizers.
        
        Returns:
            List of tuples containing (key, state_dict) for each optimizer
        """
        state_dicts = []
        for key in self.keys:
            try:
                state_dicts.append((key, self.optimizers[key].state_dict()))
            except Exception as e:
                logger.error(f"Failed to get state dict for optimizer '{key}': {e}")
        
        return state_dicts
    
    def load_state_dict(self, state_dict: List[Tuple[str, Dict[str, Any]]]) -> None:
        """
        Load state dictionary for all optimizers.
        
        Args:
            state_dict: List of tuples containing (key, state_dict) for each optimizer
        """
        loaded_keys = []
        failed_keys = []
        
        for key, state in state_dict:
            if key not in self.optimizers:
                logger.warning(f"Optimizer '{key}' not found in current optimizers")
                failed_keys.append(key)
                continue
                
            try:
                self.optimizers[key].load_state_dict(state)
                loaded_keys.append(key)
            except Exception as e:
                logger.error(f"Failed to load state dict for optimizer '{key}': {e}")
                failed_keys.append(key)
        
        logger.info(f"Successfully loaded: {loaded_keys}")
        if failed_keys:
            logger.warning(f"Failed to load: {failed_keys}")
    
    def step(self, key: Optional[str] = None, scaler: Optional[Any] = None) -> List[Any]:
        """
        Perform optimization step for specified optimizer(s).
        
        Args:
            key: Specific optimizer key to step. If None, steps all optimizers
            scaler: Optional gradient scaler for mixed precision training
            
        Returns:
            List of step results
        """
        keys = [key] if key is not None else self.keys
        
        if not keys:
            logger.warning("No optimizers to step")
            return []
        
        results = []
        for k in keys:
            try:
                result = self._step(k, scaler)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to step optimizer '{k}': {e}")
                results.append(None)
        
        return results
    
    def _step(self, key: str, scaler: Optional[Any] = None) -> Optional[Any]:
        """
        Perform single optimization step.
        
        Args:
            key: Optimizer key to step
            scaler: Optional gradient scaler for mixed precision training
            
        Returns:
            Step result or None if failed
        """
        if key not in self.optimizers:
            raise KeyError(f"Optimizer '{key}' not found")
        
        optimizer = self.optimizers[key]
        
        if scaler is not None:
            try:
                scaler.step(optimizer)
                scaler.update()
                return True
            except Exception as e:
                logger.error(f"Failed to step with scaler for '{key}': {e}")
                return None
        else:
            try:
                return optimizer.step()
            except Exception as e:
                logger.error(f"Failed to step optimizer '{key}': {e}")
                return None
    
    def zero_grad(self, key: Optional[str] = None, set_to_none: bool = False) -> None:
        """
        Zero gradients for specified optimizer(s).
        
        Args:
            key: Specific optimizer key to zero. If None, zeros all optimizers
            set_to_none: Whether to set gradients to None instead of zero
        """
        if key is not None:
            if key not in self.optimizers:
                raise KeyError(f"Optimizer '{key}' not found")
            try:
                self.optimizers[key].zero_grad(set_to_none=set_to_none)
            except Exception as e:
                logger.error(f"Failed to zero grad for optimizer '{key}': {e}")
        else:
            for k in self.keys:
                try:
                    self.optimizers[k].zero_grad(set_to_none=set_to_none)
                except Exception as e:
                    logger.error(f"Failed to zero grad for optimizer '{k}': {e}")
    
    def scheduler_step(self, *args, key: Optional[str] = None, **kwargs) -> None:
        """
        Step scheduler(s) with given arguments.
        
        Args:
            *args: Arguments to pass to scheduler.step()
            key: Specific scheduler key to step. If None, steps all schedulers
            **kwargs: Keyword arguments to pass to scheduler.step()
        """
        if key is not None:
            if key not in self.schedulers:
                logger.warning(f"Scheduler '{key}' not found")
                return
            try:
                self.schedulers[key].step(*args, **kwargs)
            except Exception as e:
                logger.error(f"Failed to step scheduler '{key}': {e}")
        else:
            for k in self.keys:
                if k in self.schedulers:
                    try:
                        self.schedulers[k].step(*args, **kwargs)
                    except Exception as e:
                        logger.error(f"Failed to step scheduler '{k}': {e}")
    
    def get_lr(self, key: Optional[str] = None) -> Union[Dict[str, List[float]], List[float]]:
        """
        Get current learning rates.
        
        Args:
            key: Specific optimizer key. If None, returns all learning rates
            
        Returns:
            Learning rates for specified optimizer or all optimizers
        """
        if key is not None:
            if key not in self.optimizers:
                raise KeyError(f"Optimizer '{key}' not found")
            return [group['lr'] for group in self.optimizers[key].param_groups]
        else:
            return {k: [group['lr'] for group in self.optimizers[k].param_groups] 
                    for k in self.keys}
    
    def __len__(self) -> int:
        """Return number of optimizers."""
        return len(self.optimizers)
    
    def __contains__(self, key: str) -> bool:
        """Check if optimizer key exists."""
        return key in self.optimizers
    
    def __repr__(self) -> str:
        """Return string representation."""
        return f"MultiOptimizer(keys={self.keys}, num_optimizers={len(self)})"


def define_scheduler(optimizer: Optimizer, params: Dict[str, Any]) -> OneCycleLR:
    """
    Create a OneCycleLR scheduler with enhanced parameter validation.
    
    Args:
        optimizer: The optimizer to schedule
        params: Dictionary containing scheduler parameters
        
    Returns:
        Configured OneCycleLR scheduler
        
    Raises:
        ValueError: If required parameters are missing or invalid
    """
    # Default parameters with validation
    max_lr = params.get('max_lr', 2e-4)
    epochs = params.get('epochs', 200)
    steps_per_epoch = params.get('steps_per_epoch', 1000)
    pct_start = params.get('pct_start', 0.0)
    div_factor = params.get('div_factor', 1.0)
    final_div_factor = params.get('final_div_factor', 1.0)
    
    # Parameter validation
    if max_lr <= 0:
        raise ValueError(f"max_lr must be positive, got {max_lr}")
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if steps_per_epoch <= 0:
        raise ValueError(f"steps_per_epoch must be positive, got {steps_per_epoch}")
    if not 0.0 <= pct_start <= 1.0:
        raise ValueError(f"pct_start must be between 0 and 1, got {pct_start}")
    if div_factor <= 0:
        raise ValueError(f"div_factor must be positive, got {div_factor}")
    if final_div_factor <= 0:
        raise ValueError(f"final_div_factor must be positive, got {final_div_factor}")
    
    try:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            anneal_strategy=params.get('anneal_strategy', 'cos'),
            cycle_momentum=params.get('cycle_momentum', True),
            base_momentum=params.get('base_momentum', 0.85),
            max_momentum=params.get('max_momentum', 0.95),
            last_epoch=params.get('last_epoch', -1)
        )
        
        logger.info(f"Created OneCycleLR scheduler with max_lr={max_lr}, "
                   f"epochs={epochs}, steps_per_epoch={steps_per_epoch}")
        
        return scheduler
        
    except Exception as e:
        logger.error(f"Failed to create scheduler: {e}")
        raise


def build_optimizer(parameters_dict: Dict[str, List[torch.nn.Parameter]], 
                   scheduler_params_dict: Dict[str, Dict[str, Any]], 
                   lr: float = 1e-4,
                   optimizer_params: Optional[Dict[str, Any]] = None) -> MultiOptimizer:
    """
    Build a MultiOptimizer with AdamW optimizers and OneCycleLR schedulers.
    
    Args:
        parameters_dict: Dictionary mapping names to parameter lists
        scheduler_params_dict: Dictionary mapping names to scheduler parameters
        lr: Base learning rate for all optimizers
        optimizer_params: Optional dictionary of additional optimizer parameters
        
    Returns:
        Configured MultiOptimizer instance
        
    Raises:
        ValueError: If parameters are invalid
        KeyError: If required keys are missing
    """
    # Validate inputs
    if not parameters_dict:
        raise ValueError("parameters_dict cannot be empty")
    
    if lr <= 0:
        raise ValueError(f"Learning rate must be positive, got {lr}")
    
    # Default optimizer parameters
    default_optimizer_params = {
        'weight_decay': 1e-4,
        'betas': (0.0, 0.99),
        'eps': 1e-9,
        'amsgrad': False
    }
    
    if optimizer_params:
        default_optimizer_params.update(optimizer_params)
    
    # Create optimizers
    optimizers = {}
    failed_optimizers = []
    
    for key, params in parameters_dict.items():
        if not params:
            logger.warning(f"Empty parameter list for optimizer '{key}'")
            continue
            
        try:
            optimizer = AdamW(
                params, 
                lr=lr, 
                **default_optimizer_params
            )
            optimizers[key] = optimizer
            logger.info(f"Created AdamW optimizer for '{key}' with lr={lr}")
            
        except Exception as e:
            logger.error(f"Failed to create optimizer for '{key}': {e}")
            failed_optimizers.append(key)
    
    if not optimizers:
        raise RuntimeError("No optimizers were successfully created")
    
    # Create schedulers
    schedulers = {}
    failed_schedulers = []
    
    for key, optimizer in optimizers.items():
        if key not in scheduler_params_dict:
            logger.warning(f"No scheduler parameters found for optimizer '{key}'")
            continue
            
        try:
            scheduler = define_scheduler(optimizer, scheduler_params_dict[key])
            schedulers[key] = scheduler
            logger.info(f"Created scheduler for '{key}'")
            
        except Exception as e:
            logger.error(f"Failed to create scheduler for '{key}': {e}")
            failed_schedulers.append(key)
    
    # Create MultiOptimizer
    try:
        multi_optim = MultiOptimizer(optimizers, schedulers)
        
        logger.info(f"Successfully created MultiOptimizer with {len(optimizers)} optimizers "
                   f"and {len(schedulers)} schedulers")
        
        if failed_optimizers:
            logger.warning(f"Failed to create optimizers for: {failed_optimizers}")
        if failed_schedulers:
            logger.warning(f"Failed to create schedulers for: {failed_schedulers}")
        
        return multi_optim
        
    except Exception as e:
        logger.error(f"Failed to create MultiOptimizer: {e}")
        raise


# Example usage and utility functions
def create_example_multioptimizer() -> MultiOptimizer:
    """Create an example MultiOptimizer for demonstration purposes."""
    
    # Example parameters (normally these would be model parameters)
    encoder_params = [torch.randn(10, 5, requires_grad=True)]
    decoder_params = [torch.randn(5, 1, requires_grad=True)]
    
    parameters_dict = {
        'encoder': encoder_params,
        'decoder': decoder_params
    }
    
    scheduler_params_dict = {
        'encoder': {
            'max_lr': 1e-3,
            'epochs': 100,
            'steps_per_epoch': 500,
            'pct_start': 0.1
        },
        'decoder': {
            'max_lr': 2e-4,
            'epochs': 100,
            'steps_per_epoch': 500,
            'pct_start': 0.05
        }
    }
    
    return build_optimizer(parameters_dict, scheduler_params_dict, lr=1e-4)


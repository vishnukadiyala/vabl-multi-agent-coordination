"""Logging utilities for MARL experiments."""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf


class MetricLogger:
    """Logger for tracking and recording metrics."""

    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        config: Optional[DictConfig] = None,
    ):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.log_dir / f"{experiment_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        if config is not None:
            config_path = self.run_dir / "config.yaml"
            OmegaConf.save(config, config_path)

        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb_writer = SummaryWriter(str(self.run_dir / "tensorboard"))
            except ImportError:
                logging.warning("TensorBoard not available")

        if use_wandb:
            try:
                import wandb

                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=f"{experiment_name}_{timestamp}",
                    config=OmegaConf.to_container(config, resolve=True)
                    if config
                    else None,
                    dir=str(self.run_dir),
                )
            except ImportError:
                logging.warning("Wandb not available")
                self.use_wandb = False

        self._step = 0

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to all configured backends."""
        if step is not None:
            self._step = step

        if self.tb_writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, self._step)

        if self.use_wandb:
            import wandb

            wandb.log(metrics, step=self._step)

    def log_video(self, key: str, video: Any, step: Optional[int] = None) -> None:
        """Log video to backends that support it."""
        if step is not None:
            self._step = step

        if self.tb_writer:
            self.tb_writer.add_video(key, video, self._step)

        if self.use_wandb:
            import wandb

            wandb.log({key: wandb.Video(video, fps=30)}, step=self._step)

    def log_histogram(
        self, key: str, values: Any, step: Optional[int] = None
    ) -> None:
        """Log histogram to backends that support it."""
        if step is not None:
            self._step = step

        if self.tb_writer:
            self.tb_writer.add_histogram(key, values, self._step)

    def close(self) -> None:
        """Close all logging backends."""
        if self.tb_writer:
            self.tb_writer.close()

        if self.use_wandb:
            import wandb

            wandb.finish()


_logger: Optional[MetricLogger] = None


def setup_logger(config: DictConfig) -> MetricLogger:
    """Set up the global logger from config."""
    global _logger

    logging.basicConfig(
        level=getattr(logging, config.logging.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    _logger = MetricLogger(
        log_dir=config.experiment.save_dir,
        experiment_name=config.experiment.name,
        use_tensorboard=config.logging.use_tensorboard,
        use_wandb=config.logging.use_wandb,
        wandb_project=config.logging.wandb_project,
        wandb_entity=config.logging.wandb_entity,
        config=config,
    )

    return _logger


def get_logger() -> Optional[MetricLogger]:
    """Get the global logger."""
    return _logger

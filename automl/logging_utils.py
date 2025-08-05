"""Simple but effective logging system for the AutoML pipeline.

Changes 2025‑07‑25
-----------------
* Always create ONE file handler so every message – from `AutoMLLogger`
  *and* from any other module that uses `logging` – is persisted.
* Attach that handler to the root logger exactly once.
* UTF‑8 encoding for complete Unicode safety.
"""

from __future__ import annotations
import os
import logging
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


class SafeStreamHandler(logging.StreamHandler):
    """Write to console but substitute characters the current
    code‑page cannot encode with '?' so we never raise."""

    def emit(self, record):
        try:
            super().emit(record)
        except UnicodeEncodeError:
            msg = self.format(record)
            enc = self.stream.encoding or "utf‑8"
            safe = msg.encode(enc, errors="replace").decode(enc)
            self.stream.write(safe + self.terminator)
            self.flush()


# ───────────────────────────────────────────────────────────────
# Helper 2: File handler that silently re‑opens if another
#           library or child process closed the log file.
# ───────────────────────────────────────────────────────────────
class ResilientFileHandler(logging.FileHandler):
    """FileHandler that reopens the log file in append mode if it's closed."""

    def __init__(self, filename, mode="a", encoding="utf-8", delay=False):
        # Force append mode to avoid log loss
        super().__init__(filename, mode, encoding, delay)

    def emit(self, record):
        if self.stream is None or self.stream.closed:
            self._reopen_append()
        try:
            super().emit(record)
        except (ValueError, OSError):
            # Stream invalid mid-write → reopen once, retry
            self._reopen_append()
            try:
                super().emit(record)
            except Exception:
                pass

    def _reopen_append(self):
        """Reopen the log file in append mode to preserve previous logs."""
        self.baseFilename = os.fspath(self.baseFilename)  # ensure str
        self.stream = open(self.baseFilename, "a", encoding=self.encoding)


class AutoMLLogger:
    """Structured logger that writes to both terminal and a UTF-8 log file."""

    def __init__(
        self,
        log_dir: Path | str | None = None,
        experiment_name: str | None = None,
        log_level: str = "INFO",
    ):
        # Setup paths
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name or f"automl_{datetime.now():%Y%m%d_%H%M%S}"
        log_file = self.log_dir / f"{self.experiment_name}.log"

        # Create logger
        self.logger = logging.getLogger(f"automl.{self.experiment_name}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        # Console output
        console_handler = SafeStreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", "%H:%M:%S")
        )
        self.logger.addHandler(console_handler)

        # File handler for this logger
        file_handler = ResilientFileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
        )
        self.logger.addHandler(file_handler)

        # Add a **separate** file handler to the root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        already = any(
            isinstance(h, ResilientFileHandler)
            and getattr(h, "_automl_tag", None) == self.experiment_name
            for h in root_logger.handlers
        )
        if not already:
            root_fh = ResilientFileHandler(log_file, mode="a", encoding="utf-8")
            root_fh.setLevel(logging.DEBUG)
            root_fh.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    "%Y-%m-%d %H:%M:%S",
                )
            )
            root_fh._automl_tag = self.experiment_name  # avoid duplicates
            root_logger.addHandler(root_fh)

        # Book-keeping
        self.start_time = time.time()
        self.pipeline_state: Dict[str, Any] = {
            "experiment_name": self.experiment_name,
            "start_time": datetime.now().isoformat(),
            "current_stage": None,
            "dataset_info": {},
            "meta_features": {},
            "rl_decisions": [],
            "bohb_results": [],
            "final_results": {},
        }

        self.logger.info(f"AutoML run started: {self.experiment_name}")

    def log_stage_start(self, stage_name: str, details: Dict[str, Any] = None):
        """Log the start of a pipeline stage."""
        self.pipeline_state["current_stage"] = stage_name

        msg = f"STAGE START: {stage_name}"
        if details:
            msg += f" | {self._format_dict(details)}"

        self.logger.info(msg)

    def log_stage_end(
        self, stage_name: str, results: Dict[str, Any] = None, elapsed_time: float = None
    ):
        """Log the end of a pipeline stage."""
        msg = f"STAGE COMPLETE: {stage_name}"

        if elapsed_time:
            msg += f" | Time: {elapsed_time:.2f}s"

        if results:
            msg += f" | {self._format_dict(results)}"

        self.logger.info(msg)

    def log_dataset_info(self, dataset_name: str, dataset_info: Dict[str, Any]):
        """Log dataset information."""
        self.pipeline_state["dataset_info"] = {"name": dataset_name, **dataset_info}

        self.logger.info(f"DATASET: {dataset_name} | {self._format_dict(dataset_info)}")

    def log_meta_features(self, meta_features: Dict[str, float], top_n: int = 5):
        """Log extracted meta-features."""
        self.pipeline_state["meta_features"] = meta_features

        # Log top features for readability
        if "feature_ranking" in meta_features:
            top_features = {}
            ranking = meta_features["feature_ranking"][:top_n]
            for feature in ranking:
                if feature in meta_features.get("normalized_features", {}):
                    top_features[feature] = round(meta_features["normalized_features"][feature], 4)
        else:
            # Fallback: show first few features
            top_features = dict(list(meta_features.items())[:top_n])
            top_features = {k: round(v, 4) for k, v in top_features.items()}

        self.logger.info(f"META-FEATURES: {self._format_dict(top_features)} ...")

    def log_rl_decision(
        self,
        episode: int,
        state: Dict[str, float],
        action: int,
        action_name: str,
        reward: float,
        details: Dict[str, Any] = None,
    ):
        """Log RL agent decision."""
        decision = {
            "episode": episode,
            "action": action,
            "action_name": action_name,
            "reward": round(reward, 4),
            "timestamp": datetime.now().isoformat(),
        }

        if details:
            decision.update(details)

        self.pipeline_state["rl_decisions"].append(decision)

        msg = f"RL DECISION #{episode}: {action_name} (action={action}) | Reward: {reward:.4f}"
        if details:
            msg += f" | {self._format_dict(details)}"

        self.logger.info(msg)

    def log_bohb_iteration(
        self,
        iteration: int,
        model_type: str,
        hyperparams: Dict[str, Any],
        performance: float,
        fidelity: str = "low",
    ):
        """Log BOHB optimization iteration."""
        result = {
            "iteration": iteration,
            "model_type": model_type,
            "hyperparams": hyperparams,
            "performance": round(performance, 4),
            "fidelity": fidelity,
            "timestamp": datetime.now().isoformat(),
        }

        self.pipeline_state["bohb_results"].append(result)

        # Simplify hyperparams for logging
        simple_params = {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in hyperparams.items()
            if k in ["lr", "batch_size", "epochs", "hidden_dim"]
        }

        self.logger.info(
            f"🔧 BOHB #{iteration} ({fidelity}): {model_type} | "
            f"Performance: {performance:.4f} | {self._format_dict(simple_params)}"
        )

    def log_final_results(
        self,
        best_model: str,
        best_hyperparams: Dict[str, Any],
        final_performance: float,
        test_predictions_saved: bool = False,
    ):
        """Log final pipeline results."""
        results = {
            "best_model": best_model,
            "best_hyperparams": best_hyperparams,
            "final_performance": round(final_performance, 4),
            "test_predictions_saved": test_predictions_saved,
            "total_time": round(time.time() - self.start_time, 2),
            "timestamp": datetime.now().isoformat(),
        }

        self.pipeline_state["final_results"] = results

        # Simplify hyperparams for logging
        simple_params = {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in best_hyperparams.items()
            if k in ["lr", "batch_size", "epochs", "hidden_dim"]
        }

        self.logger.info(
            f"FINAL RESULTS: {best_model} | "
            f"Performance: {final_performance:.4f} | "
            f"Time: {results['total_time']}s | "
            f"Params: {self._format_dict(simple_params)}"
        )

        if test_predictions_saved:
            self.logger.info("Test predictions saved successfully")

    def log_warning(self, message: str):
        """Log a warning message."""
        self.logger.warning(f" {message}")

    def log_error(self, message: str, exception: Exception = None):
        """Log an error message."""
        msg = f"ERROR: {message}"
        if exception:
            msg += f" | {type(exception).__name__}: {str(exception)}"
        self.logger.error(msg)

    def log_debug(self, message: str, data: Dict[str, Any] = None):
        """Log debug information."""
        msg = f"DEBUG: {message}"
        if data:
            msg += f" | {self._format_dict(data)}"
        self.logger.debug(msg)

    def log_terminal_output(self, output: str, source: str = "terminal"):
        """Log terminal output to the log file.

        Args:
            output: The output from the terminal to log
            source: Source of the output (e.g., 'terminal', 'stdout', 'stderr')
        """
        if not output.strip():
            return

        # Add prefix to each line to identify terminal output
        lines = output.strip().split("\n")
        for line in lines:
            if line.strip():
                self.logger.info(f"[{source}] {line}")

    def save_experiment_summary(self):
        """Save a JSON summary of the entire experiment."""
        if self.log_dir:
            summary_file = self.log_dir / f"{self.experiment_name}_summary.json"

            # Add final timing
            self.pipeline_state["total_runtime_seconds"] = round(time.time() - self.start_time, 2)
            self.pipeline_state["end_time"] = datetime.now().isoformat()

            with open(summary_file, "w") as f:
                json.dump(self.pipeline_state, f, indent=2, default=str)

            self.logger.info(f"Experiment summary saved: {summary_file}")
            return summary_file
        return None

    def _format_dict(self, data: Dict[str, Any], max_items: int = 3) -> str:
        """Format dictionary for compact logging."""
        if not data:
            return ""

        # Show only first few items to keep logs readable
        items = list(data.items())[:max_items]
        formatted_items = []

        for k, v in items:
            if isinstance(v, float):
                formatted_items.append(f"{k}={v:.4f}")
            elif isinstance(v, (int, str, bool)):
                formatted_items.append(f"{k}={v}")
            else:
                formatted_items.append(f"{k}={str(v)[:20]}...")

        result = ", ".join(formatted_items)
        if len(data) > max_items:
            result += f" ... (+{len(data) - max_items} more)"

        return result

    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get current experiment summary."""
        summary = self.pipeline_state.copy()
        summary["current_runtime_seconds"] = round(time.time() - self.start_time, 2)
        return summary


# Convenience function to create logger
def create_automl_logger(
    dataset_name: str, log_dir: str = "logs", log_level: str = "INFO"
) -> AutoMLLogger:
    """Create an AutoML logger for a specific dataset experiment.

    Args:
        dataset_name: Name of the dataset being processed
        log_dir: Directory for log files
        log_level: Logging verbosity level

    Returns:
        Configured AutoMLLogger instance
    """
    experiment_name = f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return AutoMLLogger(log_dir=Path(log_dir), experiment_name=experiment_name, log_level=log_level)


# Context manager for stage timing
class LoggedStage:
    """Context manager for automatic stage timing and logging."""

    def __init__(self, logger: AutoMLLogger, stage_name: str, details: Dict[str, Any] = None):
        self.logger = logger
        self.stage_name = stage_name
        self.details = details or {}
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.logger.log_stage_start(self.stage_name, self.details)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time

        if exc_type is not None:
            self.logger.log_error(f"Stage {self.stage_name} failed", exc_val)
        else:
            self.logger.log_stage_end(self.stage_name, elapsed_time=elapsed)

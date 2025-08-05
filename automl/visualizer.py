# automl/visualizer.py
"""
Visual Creator for AutoML Pipeline Results
==========================================
"""

import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path
from typing import Dict, Any
import warnings
from automl.constants import FEATURE_ORDER

warnings.filterwarnings("ignore")

# Set style for professional plots - academic paper standard
plt.style.use("classic")
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["axes.grid"] = True
sns.set_palette("husl")


class AutoMLVisualizer:
    """Creates visualizations for the AutoML pipeline results."""

    def __init__(
        self, results: Dict[str, Any], output_dir: Path, style: str = "paper", dpi: int = 150
    ):
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.style = style
        self.dpi = dpi

        # Set figure parameters based on style
        if style == "poster":
            self.figsize_large = (12, 8)
            self.fontsize = 14
        else:  # paper
            self.figsize_large = (10, 6)
            self.fontsize = 12

        plt.rcParams.update({"font.size": self.fontsize})

    def create_all_visuals(self):
        """Create all required visuals."""
        print("Creating AutoML Pipeline Visualizations...")
        print(f" Output directory: {self.output_dir}")
        self.create_bohb_timeline()
        self.create_meta_feature_radar_chart()
        self.create_expert_committee_dashboard()
        self.create_bohb_optimization_progress()
        print("All visualizations completed!")

    def create_bohb_timeline(self):
        """BOHB optimization timeline - separate subplot for each test dataset with consistent colors"""
        print("Creating BOHB optimization timeline...")

        detailed_logs = self.results.get("detailed_logs", [])
        if not detailed_logs:
            print("No detailed logs found")
            return

        # First, organize data by fold and dataset
        fold_dataset_data = {}
        all_datasets = set()
        all_folds = set()

        for log_entry in detailed_logs:
            dataset = log_entry.get("dataset", "unknown")
            fold = log_entry.get("fold", 0)
            iteration = log_entry.get("iteration", 0)
            model_type = log_entry.get("model_type", "unknown")

            # Get BOHB data
            bohb_data = log_entry.get("bohb", {})
            best_score = bohb_data.get("best_score", log_entry.get("reward", 0))

            all_datasets.add(dataset)
            all_folds.add(fold)

            if fold not in fold_dataset_data:
                fold_dataset_data[fold] = {}
            if dataset not in fold_dataset_data[fold]:
                fold_dataset_data[fold][dataset] = []

            fold_dataset_data[fold][dataset].append(
                {"iteration": iteration, "model_type": model_type, "score": best_score}
            )

        if not fold_dataset_data:
            print("No data found")
            return

        # Sort datasets and folds for consistency
        sorted_datasets = sorted(all_datasets)
        sorted_folds = sorted(all_folds)
        n_folds = len(sorted_folds)

        # Create consistent dataset colors
        import matplotlib.colors as mcolors

        def color_distance(color1, color2):
            """Calculate Euclidean distance between two colors in RGB space"""
            rgb1 = mcolors.to_rgb(color1)
            rgb2 = mcolors.to_rgb(color2)
            return sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)) ** 0.5

        # Model colors for complexity decisions
        model_colors = {"simple": "#1f77b4", "medium": "#ff7f0e", "complex": "#d62728"}

        # Generate dataset colors with minimum distance from model colors
        all_colors = list(plt.cm.tab10.colors)
        selected_dataset_colors = []
        min_distance = 0.3

        for color in all_colors:
            if all(
                color_distance(color, model_color) > min_distance
                for model_color in model_colors.values()
            ):
                selected_dataset_colors.append(color)
            if len(selected_dataset_colors) >= len(sorted_datasets):
                break

        # Ensure we have enough colors
        selected_dataset_colors = selected_dataset_colors[: len(sorted_datasets)]

        # Create consistent dataset colors dictionary
        dataset_colors = {}
        for i, dataset in enumerate(sorted_datasets):
            dataset_colors[dataset] = selected_dataset_colors[i % len(selected_dataset_colors)]

        # Create figure with subplots - one for each fold
        fig, axes = plt.subplots(1, n_folds, figsize=(4 * n_folds, 6), sharey=True)

        # Handle case where there's only one fold
        if n_folds == 1:
            axes = [axes]

        all_scores = []

        # Plot each fold in its own subplot
        for fold_idx, fold in enumerate(sorted_folds):
            ax = axes[fold_idx]

            # Extract test dataset from fold name (cv_amazon -> amazon)
            test_dataset = fold.replace("cv_", "") if fold.startswith("cv_") else fold

            # Plot each dataset for this fold
            for dataset in sorted_datasets:
                if dataset not in fold_dataset_data[fold]:
                    continue

                data = fold_dataset_data[fold][dataset]
                df = pd.DataFrame(data)
                df = df.sort_values("iteration")

                x_vals = df["iteration"].values
                y_vals = df["score"].values
                color = dataset_colors[dataset]

                # Collect all scores for y-axis scaling
                all_scores.extend(y_vals)

                # Plot the optimization progress line (all training datasets get solid lines)
                ax.plot(
                    x_vals,
                    y_vals,
                    "-",
                    linewidth=2,
                    alpha=0.8,
                    color=color,
                    label=dataset if fold_idx == 0 else "",
                )

                # Add small dots for all trials
                ax.scatter(x_vals, y_vals, s=20, color=color, alpha=0.4)

                # Add larger colored dots showing model complexity decisions
                for _, row in df.iterrows():
                    model_type = row["model_type"]
                    model_color = model_colors.get(model_type, "gray")

                    ax.scatter(
                        row["iteration"],
                        row["score"],
                        s=100,
                        c=model_color,
                        alpha=0.9,
                        edgecolors="white",
                        linewidth=1,
                        zorder=10,
                    )

            # Customize each subplot
            ax.set_xlabel("RL Training Iteration", fontweight="bold")
            if fold_idx == 0:
                ax.set_ylabel("Accuracy", fontweight="bold")

            # Set title showing which dataset is being tested
            ax.set_title(
                f"{test_dataset.title()} Agent\n(Trained without {test_dataset.title()} data)",
                fontweight="bold",
                pad=10,
                fontsize=12,
            )

            ax.grid(True, alpha=0.3)

            # Format y-axis as percentages
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

        # Set dynamic y-axis limits for all subplots
        if all_scores:
            min_score = min(all_scores)
            max_score = max(all_scores)

            y_range = max_score - min_score
            y_padding = max(0.02, y_range * 0.1)

            y_min = max(0, min_score - y_padding)
            y_max = min(1.0, max_score + y_padding)

            for ax in axes:
                ax.set_ylim(y_min, 1.0)

        # title with subtitle
        fig.suptitle(
            "RL+BOHB Agent Training: Leave-One-Out Cross-Validation",
            fontweight="bold",
            fontsize=16,
            y=0.98,
        )

        # Create legends (only show once, using the first subplot)
        dataset_handles = [
            plt.Line2D([0], [0], color=dataset_colors[dataset], linewidth=3, label=dataset.title())
            for dataset in sorted_datasets
        ]

        # Model decision legend
        model_handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=10,
                label=f"{model_type.title()} Model",
                linestyle="None",
            )
            for model_type, color in model_colors.items()
        ]

        leg1 = fig.legend(
            handles=dataset_handles,
            title="Training Datasets",
            bbox_to_anchor=(0.94, 0.7),
            loc="upper left",
            title_fontsize=12,
            fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=True,
            facecolor="white",
            edgecolor="gray",
        )
        leg2 = fig.legend(
            handles=model_handles,
            title="Model Complexity",
            bbox_to_anchor=(0.94, 0.4),
            loc="upper left",
            numpoints=1,
            title_fontsize=12,
            fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=True,
            facecolor="white",
            edgecolor="gray",
        )

        # Style the legend titles
        leg1.get_title().set_fontweight("bold")
        leg2.get_title().set_fontweight("bold")

        plt.tight_layout()
        plt.subplots_adjust(right=0.90)

        plt.savefig(
            self.output_dir / "bohb_optimization_timeline.png", dpi=self.dpi, bbox_inches="tight"
        )
        plt.close()

        print("BOHB optimization timeline saved")

    def create_meta_feature_radar_chart(self):
        """Create a professional radar chart showing meta-features, performance, and model selection."""
        print("Creating Advanced Meta-Feature Radar Chart...")

        # Get meta-features and results
        meta_features = self.results.get("meta_features", {})
        cv_results = self.results.get("cv_results", {})
        performance = cv_results.get("performance", {})
        folds = cv_results.get("folds", [])

        if not meta_features:
            print("No meta-features found for visualization")
            return

        # Use a colorblind-friendly palette with distinct colors
        dataset_colors = plt.cm.Set1.colors

        # Select top features that have variation and data
        def select_top_features(meta_features_dict, total_features=FEATURE_ORDER, max_features=10):
            """Select top features with variation."""
            valid_features = []
            for feature in total_features:
                # Collect values for this feature across all datasets
                values = [
                    meta_features_dict[dataset].get(feature, 0) for dataset in meta_features_dict
                ]

                # Remove NaN or zero values
                values = [v for v in values if v is not None and v != 0]

                # Check if feature has variation
                if len(set(values)) > 1:
                    valid_features.append(feature)

                    # Stop if we've reached max features
                    if len(valid_features) >= max_features:
                        break

            return valid_features

        # Select top features respecting FEATURE_ORDER
        top_features = select_top_features(meta_features)

        # Create figure with high-quality setup
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(
            figsize=(16, 10), subplot_kw=dict(projection="polar"), facecolor="white"
        )

        # Angle for each feature
        angles = [n / float(len(top_features)) * 2 * np.pi for n in range(len(top_features))]
        angles += angles[:1]  # Complete the circle

        # Prepare datasets and their model selections
        dataset_models = {fold["held_out"]: fold["selected_model"] for fold in folds}

        # Collect ALL values for each feature across ALL datasets for proper normalization
        feature_ranges = {}
        for feature in top_features:
            all_values = []
            for dataset in meta_features:
                val = meta_features[dataset].get(feature, 0)
                # Apply log transform if needed
                val = np.log1p(val) if val > 1000 else val
                all_values.append(val)

            min_val = min(all_values)
            max_val = max(all_values)
            feature_ranges[feature] = (min_val, max_val)

        def normalize_feature(feature_values, feature_names):
            """Normalize using global min/max for each feature across all datasets."""
            normalized = []
            for i, val in enumerate(feature_values):
                feature = feature_names[i]
                min_val, max_val = feature_ranges[feature]

                if max_val == min_val:
                    normalized.append(0.5)
                else:
                    normalized.append((val - min_val) / (max_val - min_val))

            return normalized

        # Normalize feature values
        # Plot for each dataset
        for i, (dataset, features) in enumerate(meta_features.items()):
            # Collect feature values
            feature_values = [features.get(feature, 0) for feature in top_features]

            # Logarithmic transformation for large-scale features
            feature_values = [np.log1p(val) if val > 1000 else val for val in feature_values]

            # Normalize values
            # Normalize values using global ranges
            normalized_values = normalize_feature(feature_values, top_features)
            normalized_values += normalized_values[:1]

            # Get model type
            model_type = dataset_models.get(dataset, "unknown").lower()

            # Use a unique color for each dataset
            dataset_color = dataset_colors[i % len(dataset_colors)]

            # Plot with gradient and more professional styling
            ax.plot(
                angles,
                normalized_values,
                linewidth=4,
                linestyle="solid",
                label=f"{dataset.title()}",
                color=dataset_color,
                alpha=0.7,
            )

            # Fill with transparent gradient
            ax.fill(angles, normalized_values, color=dataset_color, alpha=0.1)

        # Styling improvements
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Draw axis lines for each angle and label
        ax.set_thetagrids(
            np.degrees(angles[:-1]),
            [f.replace("_", " ").title() for f in top_features],
            fontweight="bold",
        )

        # Create custom legend
        # Prepare legend entries
        legend_entries = []
        for dataset, color in zip(meta_features.keys(), dataset_colors):
            # Get model and performance
            model_type = dataset_models.get(dataset, "unknown").lower()
            perf = performance.get(dataset, 0)

            # Create a custom legend entry
            legend_entry = f"{dataset.title()} → {model_type.title()} ({perf*100:.0f}%)"
            legend_entries.append(plt.Line2D([0], [0], color=color, lw=4, label=legend_entry))

        # Professional legend with dataset details
        plt.legend(
            handles=legend_entries,
            loc="center left",
            bbox_to_anchor=(1.05, 0.5),
            title="Datasets, Models & Performance",
            title_fontsize=13,
            frameon=True,
            fancybox=True,
            shadow=True,
            edgecolor="lightgray",  # Softer border
            facecolor="white",
        )

        # Title
        plt.title(
            f"Key Meta-Features That Drive Model Type Selection",
            fontweight="bold",
            fontsize=16,
            y=1.08,
        )

        # Background and grid improvements
        ax.grid(color="lightgray", linestyle="--", linewidth=0.5)
        plt.tight_layout()

        # Save high-quality figures
        plt.savefig(
            self.output_dir / "meta_feature_radar_chart.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close()

        print("Advanced Meta-Feature Radar Chart saved")

    def create_expert_committee_dashboard(self):
        """Expert Committee Performance Dashboard - horizontal bar chart showing final model selection"""
        print("Creating Expert Committee Performance Dashboard...")

        final_selections = self.results.get("final_selections", {})
        if not final_selections:
            print("No final selections found")
            return

        # Model complexity colors
        model_colors = {"simple": "#1f77b4", "medium": "#ff7f0e", "complex": "#d62728"}

        # Prepare data for visualization
        agents = []
        performances = []
        model_types = []
        configs = []

        for agent_name, selection_info in final_selections.items():
            agents.append(agent_name.replace("cv_", "").title() + "\nAgent")
            performances.append(selection_info.get("bohb_score", 0) * 100)
            model_types.append(selection_info.get("model_type", "unknown"))

            # Extract key hyperparameters for annotation
            config = selection_info.get("best_config", {})
            config_str = self._format_config_dynamic(config, max_length=60)
            configs.append(config_str)

        if not agents:
            print("No agent data found")
            return

        # Find the best performing agent
        best_idx = np.argmax(performances)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create horizontal bars
        y_positions = np.arange(len(agents))
        bars = []

        for i, (agent, perf, model_type) in enumerate(zip(agents, performances, model_types)):
            color = model_colors.get(model_type, "#gray")

            # Create bar with special styling for winner
            if i == best_idx:
                # Winner gets bold border and slightly different styling
                bar = ax.barh(
                    y_positions[i],
                    perf,
                    color=color,
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=3,
                    height=0.6,
                )
            else:
                bar = ax.barh(
                    y_positions[i],
                    perf,
                    color=color,
                    alpha=0.7,
                    edgecolor="white",
                    linewidth=1,
                    height=0.6,
                )
            bars.append(bar)

        # Customize the plot
        ax.set_yticks(y_positions)
        ax.set_yticklabels(agents)
        ax.set_xlabel("Accuracy", fontweight="bold", fontsize=14)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100.0, decimals=0))
        ax.set_ylabel("CV Agents", fontweight="bold", fontsize=14)

        # Set x-axis limits with some padding
        max_perf = max(performances)
        ax.set_xlim(0, max_perf + 10)

        # Add performance labels on bars
        for i, (bar, perf, model_type) in enumerate(zip(bars, performances, model_types)):
            # Performance percentage
            ax.text(
                perf + 0.5,
                bar[0].get_y() + bar[0].get_height() / 2,
                f"{perf:.1f}%",
                va="center",
                ha="left",
                fontweight="bold",
                fontsize=11,
            )

            # Model type label
            ax.text(
                perf / 2,
                bar[0].get_y() + bar[0].get_height() / 2,
                f"{model_type.title()}",
                va="center",
                ha="center",
                fontweight="bold",
                color="white",
                fontsize=12,
            )

        # Add "SELECTED" annotation for winner
        winner_bar = bars[best_idx]
        ax.annotate(
            "SELECTED",
            xy=(
                performances[best_idx] + 0.3,
                winner_bar[0].get_y() + winner_bar[0].get_height() / 2 + 0.2,
            ),
            xytext=(
                performances[best_idx] + 6,
                winner_bar[0].get_y() + winner_bar[0].get_height() + 0.2,
            ),  # Move up
            va="center",
            ha="center",
            fontweight="bold",
            fontsize=11,
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="#27ae60",
                alpha=0.95,
                edgecolor="#1e8449",
                linewidth=2,
            ),
            arrowprops=dict(
                arrowstyle="->",
                connectionstyle="arc3,rad=-0.2",  # Curve down to bar
                color="#27ae60",
                lw=2,
            ),
        )

        winner_config = configs[best_idx]
        ax.text(
            performances[best_idx] / 2,
            winner_bar[0].get_y() + 0.1,
            winner_config,
            va="center",
            ha="center",
            fontsize=10,
            color="white",
            weight="bold",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="#2c3e50",
                alpha=0.9,  # Professional dark blue-gray
                edgecolor="white",
                linewidth=1,
            ),
        )

        # Set title
        ax.set_title(
            "Final Phase: CV Agents Performances → Best Agent Selection",
            fontweight="bold",
            fontsize=16,
            pad=20,
        )

        # Save the plot
        plt.savefig(
            self.output_dir / "expert_committee_dashboard.png", dpi=self.dpi, bbox_inches="tight"
        )
        plt.close()

        print("Expert Committee Performance Dashboard saved")

    def _format_config_dynamic(self, config: Dict[str, Any], max_length: int = 40) -> str:
        """Dynamically format configuration dictionary with intelligent truncation"""
        if not config:
            return "Default parameters"

        try:
            # Smart formatting for different value types
            formatted_params = []

            for key, value in config.items():
                if isinstance(value, float):
                    # Limit floats to 2 decimal places
                    if value < 0.001:
                        param_str = f"{key}={value:.1e}"  # Scientific notation for very small
                    elif value > 1000:
                        param_str = f"{key}={value:.0f}"  # No decimals for large numbers
                    else:
                        param_str = f"{key}={value:.2f}"
                elif isinstance(value, int):
                    # Format large integers with commas
                    if value >= 1000:
                        param_str = f"{key}={value:,}"
                    else:
                        param_str = f"{key}={value}"
                elif isinstance(value, str):
                    # Truncate long strings
                    short_val = value[:8] + "..." if len(value) > 8 else value
                    param_str = f"{key}={short_val}"
                elif isinstance(value, bool):
                    param_str = f"{key}={str(value).lower()}"
                else:
                    # Fallback for other types
                    param_str = f"{key}={str(value)[:6]}"

                formatted_params.append(param_str)

            # Join parameters
            config_str = ", ".join(formatted_params)

            # Intelligent truncation if too long
            if len(config_str) <= max_length:
                return config_str
            else:
                # Build string within length limit
                result_parts = []
                current_length = 0

                for param in formatted_params:
                    if (
                        current_length + len(param) + 2 <= max_length - 3
                    ):  # Reserve 3 chars for "..."
                        result_parts.append(param)
                        current_length += len(param) + 2  # +2 for ", "
                    else:
                        break

                if len(result_parts) < len(formatted_params):
                    return ", ".join(result_parts) + "..."
                else:
                    return ", ".join(result_parts)

        except Exception as e:
            print(f"Error formatting config: {e}")
            try:
                # Just convert the whole dict to string and truncate
                config_str = str(config).replace("'", "").replace("{", "").replace("}", "")
                return (
                    config_str[: max_length - 3] + "..."
                    if len(config_str) > max_length
                    else config_str
                )
            except Exception as e2:
                print(f"Error in fallback formatting: {e2}")
                return "Configuration available"

    def create_bohb_optimization_progress(self):
        """Create BOHB optimization progress visualization from stored results"""
        print("Creating BOHB optimization progress visualization...")

        bohb_evaluations = self.results.get("bohb_evaluations", [])

        if not bohb_evaluations:
            print("No BOHB evaluations found in results")
            return

        # Group by model type and create sessions
        bohb_sessions = {}

        for eval_data in bohb_evaluations:
            model_type = eval_data["model_type"]
            dataset = eval_data["dataset"]
            cv_fold = eval_data["cv_fold"]

            session_key = f"{cv_fold}_{dataset}_{model_type}"

            if session_key not in bohb_sessions:
                bohb_sessions[session_key] = {
                    "dataset": dataset,
                    "model_type": model_type,
                    "fold": cv_fold,
                    "trials": [],
                }

            # Get the individual trials from full_optimization_history
            full_history = eval_data.get("full_optimization_history", [])
            for trial in full_history:
                bohb_sessions[session_key]["trials"].append(trial)

        if not bohb_sessions:
            print("No BOHB sessions found in evaluations")
            return

        print(f"Found {len(bohb_sessions)} BOHB sessions from evaluations")
        self._create_bohb_visualization(bohb_sessions)

    def _create_bohb_visualization(self, bohb_sessions):
        """Create BOHB visualization with subplots per model type and correct sequential order"""
        # Group sessions by model type
        sessions_by_model = {}
        for session_key, session_data in bohb_sessions.items():
            model_type = session_data["model_type"]
            if model_type not in sessions_by_model:
                sessions_by_model[model_type] = []
            sessions_by_model[model_type].append((session_key, session_data))

        n_models = len(sessions_by_model)
        if n_models == 0:
            print("No model types found")
            return

        # Create figure with subplots per model type
        fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 6))
        if n_models == 1:
            axes = [axes]

        # Color schemes
        model_colors = {"simple": "#1f77b4", "medium": "#ff7f0e", "complex": "#d62728"}

        # Collect all unique budgets across all model types
        all_unique_budgets = set()
        for model_type, sessions in sessions_by_model.items():
            for session_key, session_data in sessions[:6]:
                trials = session_data["trials"]
                for trial in trials:
                    all_unique_budgets.add(trial["budget"])

        all_unique_budgets = sorted(all_unique_budgets)
        print(f"All budgets across all models: {all_unique_budgets}")

        for idx, (model_type, sessions) in enumerate(sorted(sessions_by_model.items())):
            ax = axes[idx] if n_models > 1 else axes[0]

            # Limit to first 6 sessions for better visualization
            max_sessions = 5
            if len(sessions) > max_sessions:
                sessions = sessions[:max_sessions]

            # Collect trials from all sessions for this model type with global sequential order
            all_trials_with_global_order = []
            global_eval_counter = 0
            session_boundaries = []

            # Process each session sequentially (now limited to 6)
            for session_key, session_data in sessions:
                trials = session_data["trials"]
                if not trials:
                    continue

                # Mark session boundary
                session_boundaries.append(global_eval_counter + 1)

                # Sort trials within this session by their local evaluation number
                trials.sort(key=lambda x: x["evaluation"])

                # Add each trial with global sequential numbering
                for trial in trials:
                    global_eval_counter += 1
                    trial_copy = trial.copy()
                    trial_copy["global_evaluation"] = global_eval_counter
                    trial_copy["session"] = session_key  # Add session information
                    all_trials_with_global_order.append(trial_copy)

            if not all_trials_with_global_order:
                continue

            # Extract unique budgets and sort them
            from matplotlib.colors import LinearSegmentedColormap
            import numpy as np

            # Create a smooth blue-to-red colormap
            def create_blue_to_red_colormap():
                """Create a smooth blue to red colormap via white/light colors"""
                colors = [
                    "#1E88E5",  # Strong Blue (fast/cheap)
                    "#42A5F5",  # Light Blue
                    "#90CAF9",  # Very Light Blue
                    "#FFECB3",  # Light Yellow (neutral)
                    "#FFB74D",  # Light Orange
                    "#FF7043",  # Orange
                    "#E53935",  # Strong Red (expensive/slow)
                ]
                return LinearSegmentedColormap.from_list("blue_to_red", colors, N=256)

            # Replace your colormap section with this:
            # Extract unique budgets and sort them
            all_budgets = [t["budget"] for t in all_trials_with_global_order]
            unique_budgets = sorted(set(all_budgets))
            min_budget = min(unique_budgets)
            max_budget = max(unique_budgets)

            print(
                f"Found budgets for {model_type}: {unique_budgets} (min: {min_budget}, max: {max_budget})"
            )

            # Create smooth blue-to-red colormap
            budget_colormap = create_blue_to_red_colormap()
            budget_colors = {}
            budget_sizes = {}
            budget_alphas = {}

            for budget in unique_budgets:
                if len(unique_budgets) == 1:
                    # Single budget - use middle blue color
                    norm_budget = 0.3  # Slightly blue
                else:
                    # Multiple budgets - spread across full range
                    norm_budget = (budget - min_budget) / (max_budget - min_budget)

                budget_colors[budget] = budget_colormap(norm_budget)

                # Size scaling: 50-120 range
                if max_budget > min_budget:
                    budget_sizes[budget] = 50 + (budget - min_budget) * 70 / (
                        max_budget - min_budget
                    )
                else:
                    budget_sizes[budget] = 85  # Medium size for single budget

                # Alpha scaling: 0.6-0.9 range
                if max_budget > min_budget:
                    budget_alphas[budget] = 0.6 + (budget - min_budget) * 0.3 / (
                        max_budget - min_budget
                    )
                else:
                    budget_alphas[budget] = 0.75  # Medium alpha for single budget
            # Extract data for plotting
            global_evaluations = [t["global_evaluation"] for t in all_trials_with_global_order]
            scores = [t["score"] for t in all_trials_with_global_order]

            # Assign colors, sizes, and alphas dynamically
            colors = []
            sizes = []
            alphas = []

            for trial in all_trials_with_global_order:
                budget = trial["budget"]
                colors.append(budget_colors[budget])
                sizes.append(budget_sizes[budget])
                alphas.append(budget_alphas[budget])

            # Plot all points
            ax.scatter(
                global_evaluations,
                scores,
                c=colors,
                s=sizes,
                alpha=alphas,
                edgecolors="white",
                linewidth=1,
                zorder=5,
            )

            # Calculate and plot "best so far" line PER SESSION (resets at boundaries)
            current_session = None
            session_best = 0
            running_best = []

            for trial in all_trials_with_global_order:
                # Reset best score when we enter a new session
                if trial["session"] != current_session:
                    session_best = 0  # Reset for new session
                    current_session = trial["session"]

                # Update best within this session
                session_best = max(session_best, trial["score"])
                running_best.append(session_best)

            # Plot the per-session best line (separate segments, no connection between sessions)
            current_session = None
            session_start_idx = 0

            for i, trial in enumerate(all_trials_with_global_order):
                # When we hit a new session, plot the previous session's line
                if trial["session"] != current_session:
                    if current_session is not None:  # Not the first session
                        # Plot line for previous session
                        session_x = global_evaluations[session_start_idx:i]
                        session_y = running_best[session_start_idx:i]
                        ax.plot(session_x, session_y, "#2E7D2E", linewidth=4, alpha=0.9, zorder=10)

                    # Start new session
                    current_session = trial["session"]
                    session_start_idx = i

            # Plot the last session's line
            if session_start_idx < len(global_evaluations):
                session_x = global_evaluations[session_start_idx:]
                session_y = running_best[session_start_idx:]
                ax.plot(session_x, session_y, "#2E7D2E", linewidth=4, alpha=0.9, zorder=10)

            # Add label only once (to avoid duplicate legend entries)
            ax.plot([], [], "#2E7D2E", linewidth=4, alpha=0.9, label="Running Best (Per Session)")

            # Add session boundaries (vertical lines only, no labels)
            for boundary in session_boundaries[1:]:  # Skip first boundary
                ax.axvline(
                    x=boundary - 0.5, color="black", linestyle="--", alpha=0.7, linewidth=2
                )  # Changed alpha from 0.3 to 0.7, linewidth from 2 to 3

            # Customize subplot
            ax.set_title(
                f"{model_type.title()} Model: Multi-Session BOHB Optimization",
                fontweight="bold",
                color=model_colors.get(model_type, "black"),
            )
            ax.set_xlabel("Trial Number Across Independent BOHB Sessions")
            ax.set_ylabel("Model Accuracy")
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))

            # Create gradient bar for the rightmost subplot only
            if idx == len(sorted(sessions_by_model.items())) - 1:  # Last subplot
                # Create gradient bar on the right side of the last subplot
                gradient_ax = fig.add_axes([0.93, 0.55, 0.02, 0.3])  # Higher position (0.55-0.85)
                gradient = np.linspace(0, 1, 256).reshape(256, 1)

                gradient_ax.imshow(gradient, aspect="auto", cmap=budget_colormap.reversed())
                gradient_ax.set_xticks([])
                gradient_ax.set_yticks([])

                # Add labels
                gradient_ax.text(
                    1.5,
                    0.9,
                    "High Fidelity\n(More Data \n Thorough Evaluation)",
                    transform=gradient_ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=9,
                    fontweight="bold",
                )
                gradient_ax.text(
                    1.5,
                    0.1,
                    "Low Fidelity\n(Less Data \n Quick Exploration )",
                    transform=gradient_ax.transAxes,
                    va="bottom",
                    ha="left",
                    fontsize=9,
                    fontweight="bold",
                )

                # Add title for gradient bar
                gradient_ax.text(
                    0.5,
                    1.05,
                    "BOHB Budget",
                    transform=gradient_ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

        # Create single legend for all subplots
        legend_elements = []
        for budget in sorted(all_unique_budgets, reverse=True):
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=budget_colormap(
                        (budget - min(all_unique_budgets))
                        / (max(all_unique_budgets) - min(all_unique_budgets))
                        if len(all_unique_budgets) > 1
                        else 0.3
                    ),
                    markersize=8,
                    label=f"Budget {budget}",
                )
            )
        legend_elements.append(
            plt.Line2D([0], [0], color="#2E7D2E", linewidth=4, label="Best Performance Found")
        )
        legend_elements.append(
            plt.Line2D(
                [0], [0], color="black", linestyle="--", linewidth=2, label="BOHB Session Boundary"
            )
        )

        fig.legend(
            handles=legend_elements, bbox_to_anchor=(0.93, 0.45), loc="upper left", fontsize=11
        )

        # Overall title
        fig.suptitle(
            "Multi-Session BOHB: Hyperband Budget Allocation",
            fontweight="bold",
            fontsize=16,
            y=0.98,
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.85, right=0.90)

        # Save plot
        plt.savefig(
            self.output_dir / "bohb_optimization_progress.png", dpi=self.dpi, bbox_inches="tight"
        )
        plt.close()

        print("BOHB optimization progress visualization saved")


def save_all_figures(results: Dict[str, Any], output_dir: Path, **kwargs):
    """Convenience function to create all visualizations."""
    visualizer = AutoMLVisualizer(results=results, output_dir=output_dir, **kwargs)
    visualizer.create_all_visuals()


def load_results(results_path: Path) -> Dict[str, Any]:
    """Load results from pickle file."""
    print(f"Loading results from: {results_path}")

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path, "rb") as f:
        results = pickle.load(f)

    print(f"Loaded results with {len(results)} top-level keys")

    # Print summary of available data
    print("\nAvailable data:")
    for key, value in results.items():
        if isinstance(value, (list, dict)):
            print(f"  • {key}: {len(value)} items")
        else:
            print(f"  • {key}: {type(value).__name__}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Create visualizations for AutoML pipeline results"
    )
    parser.add_argument(
        "--results", type=str, required=True, help="Path to the results pickle file"
    )
    parser.add_argument(
        "--output", type=str, default="visuals", help="Output directory for visualizations"
    )
    parser.add_argument(
        "--style",
        type=str,
        choices=["paper", "poster"],
        default="paper",
        help="Style of visualizations (paper or poster)",
    )
    parser.add_argument("--dpi", type=int, default=150, help="DPI for saved images")

    args = parser.parse_args()

    # Load results
    results_path = Path(args.results)
    results = load_results(results_path)

    # Create visualizer
    visualizer = AutoMLVisualizer(
        results=results, output_dir=Path(args.output), style=args.style, dpi=args.dpi
    )

    # Create all visuals
    visualizer.create_all_visuals()

    print(f"\n All visualizations saved to: {Path(args.output).absolute()}")


if __name__ == "__main__":
    main()

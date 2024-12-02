import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend, which supports interactive features
import matplotlib.pyplot as plt
from transformers import TrainerCallback

class PlotPredictionsCallback(TrainerCallback):
    def __init__(self, plot_interval=500):
        self.plot_interval = plot_interval

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # Print overall metrics
            if 'eval_precision' in logs:
                print(f"Precision: {logs['eval_precision']:.4f}")
            if 'eval_recall' in logs:
                print(f"Recall: {logs['eval_recall']:.4f}")
            if 'eval_f1' in logs:
                print(f"F1 Score: {logs['eval_f1']:.4f}")
            
            # Print per-category metrics (if available)
            if 'per_category' in logs:
                print("Per-category Metrics:")
                for category, metrics in logs['per_category'].items():
                    print(f"  {category}:")
                    print(f"    Precision: {metrics['precision']:.4f}")
                    print(f"    Recall: {metrics['recall']:.4f}")
                    print(f"    F1 Score: {metrics['f1']:.4f}")
            
            # Plot metrics every few evaluations
            if state.global_step % self.plot_interval == 0:
                self.plot_metrics(state.global_step)

    def update_plot(self, step):
        # Update plot data
        self.train_losses.append(step)  # Placeholder, adjust according to actual losses
        self.val_losses.append(step)    # Placeholder, adjust according to actual losses
        
        # Clear previous plots
        for ax in self.axs.flat:
            ax.clear()

        # Plot Losses
        self.axs[0, 0].plot(self.train_losses, label="Training Loss")
        self.axs[0, 0].plot(self.val_losses, label="Validation Loss")
        self.axs[0, 0].set_xlabel("Steps")
        self.axs[0, 0].set_ylabel("Loss")
        self.axs[0, 0].set_title("Loss Over Time")
        self.axs[0, 0].legend()

        # Plot Precision, Recall, F1 Score
        self.axs[0, 1].plot(self.precisions, label="Precision")
        self.axs[0, 1].plot(self.recalls, label="Recall")
        self.axs[0, 1].plot(self.f1_scores, label="F1 Score")
        self.axs[0, 1].set_xlabel("Steps")
        self.axs[0, 1].set_ylabel("Score")
        self.axs[0, 1].set_title("Precision, Recall, F1 Score")
        self.axs[0, 1].legend()

        self.fig.tight_layout()
        plt.draw()
        plt.pause(0.1)  # Allow the plot to refresh and be resizable
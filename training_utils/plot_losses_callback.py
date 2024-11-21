import matplotlib.pyplot as plt
from transformers import TrainerCallback


class PlotLossesCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.steps = []

        # Initialize the plot
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.line_train, = self.ax.plot([], [], label='Training Loss')
        self.line_eval, = self.ax.plot([], [], label='Evaluation Loss')
        self.ax.set_xlabel("Steps")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Training and Evaluation Loss")
        self.ax.legend()
        plt.show()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        # Update train and eval losses
        if 'loss' in logs:
            self.train_losses.append(logs['loss'])
            self.steps.append(state.global_step)

        if 'eval_loss' in logs:
            self.eval_losses.append(logs['eval_loss'])

        # Update plot data
        self.line_train.set_xdata(self.steps)
        self.line_train.set_ydata(self.train_losses)
        if self.eval_losses:
            self.line_eval.set_xdata(self.steps[:len(self.eval_losses)])
            self.line_eval.set_ydata(self.eval_losses)

        # Rescale plot limits
        self.ax.relim()
        self.ax.autoscale_view()

        # Redraw the plot
        plt.draw()
        plt.pause(0.01)

    def on_train_end(self, args, state, control, **kwargs):
        # Finalize the plot
        plt.ioff()  # Turn off interactive mode
        plt.show()

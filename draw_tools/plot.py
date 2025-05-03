import matplotlib.pyplot as plt

colors_set = {'Kraftime': ('#E3E37D', '#968A62')}

def plot(runner, axes, set=colors_set['Kraftime']):
    train_color = set[0]
    dev_color = set[1]

    # X-axis ranges for each
    train_steps = list(range(len(runner.train_loss)))
    dev_steps = list(range(len(runner.dev_loss)))

    # Plotting training and validation losses
    axes[0].plot(train_steps, runner.train_loss, color=train_color, label="Train loss")
    axes[0].plot(dev_steps, runner.dev_loss, color=dev_color, linestyle="--", label="Dev loss")
    axes[0].set_ylabel("Loss")
    axes[0].set_xlabel("Iteration")
    axes[0].legend(loc='upper right')

    # Plotting training and validation accuracies
    axes[1].plot(train_steps, runner.train_scores, color=train_color, label="Train accuracy")
    axes[1].plot(dev_steps, runner.dev_scores, color=dev_color, linestyle="--", label="Dev accuracy")
    axes[1].set_ylabel("Score")
    axes[1].set_xlabel("Iteration")
    axes[1].legend(loc='lower right')

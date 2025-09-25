import matplotlib.pyplot as plt

def plot_curves(runs, metric_key: str = "acc"):
    for label, xs, ys in runs:
        plt.plot(xs, ys, label=label)
    plt.xlabel("Labeled samples")
    plt.ylabel(metric_key)
    plt.legend()
    plt.tight_layout()
    return plt.gcf()

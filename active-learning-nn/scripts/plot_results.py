import os, json, glob
import matplotlib.pyplot as plt

def main():
    for path in glob.glob("experiments/results/*.json"):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            logs = json.load(f)
        xs = [r["labeled"] for r in logs]
        if "acc" in logs[-1]:
            ys = [r["acc"] for r in logs]
            ylab = "Accuracy (↑)"
        else:
            ys = [-(r["loss"]**0.5) for r in logs]  # -RMSE
            ylab = "-RMSE (↑)"
        plt.figure()
        plt.plot(xs, ys, marker=".")
        plt.xlabel("Labeled samples")
        plt.ylabel(ylab)
        plt.title(name)
        plt.tight_layout()
        out = f"experiments/results/{name}.png"
        plt.savefig(out, dpi=160)
        print(f"Wrote {out}")

if __name__ == "__main__":
    main()

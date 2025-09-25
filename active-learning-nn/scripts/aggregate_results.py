import os, json, glob, numpy as np
from alnn.metrics.curves import area_under_learning_curve
from alnn.utils.logging import print_table

def main():
    rows = []
    for path in glob.glob("experiments/results/*.json"):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            logs = json.load(f)
        xs = [r["labeled"] for r in logs]

        if "acc" in logs[-1]:            # classification
            ys = [r["acc"] for r in logs]
            final_metric_name = "acc"
            final_metric = ys[-1]
            au = area_under_learning_curve(xs, ys)  # higher is better
        else:                              # regression
            rmses = [float(r["loss"])**0.5 for r in logs]      # loss is MSE
            ys = [ -rm for rm in rmses ]                       # use -RMSE so higher is better
            final_metric_name = "rmse"
            final_metric = rmses[-1]
            au = area_under_learning_curve(xs, ys)

        rows.append((name, xs[-1], final_metric_name, final_metric, au))

    print_table(rows, headers=["run", "labeled_end", "metric", "final_value", "AUCLC(-RMSE|acc)"])

if __name__ == "__main__":
    main()

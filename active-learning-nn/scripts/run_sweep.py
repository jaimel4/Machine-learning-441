import argparse, yaml, json, os, sys
from alnn.data.datasets import load_dataset, make_splits
from alnn.active_learning.loop import run_active_learning
from alnn.utils.seed import set_seed

def load_cfg(path):
    if not os.path.exists(path):
        sys.exit(f"Config not found: {path}")
    with open(path, "r") as f:
        txt = f.read().strip()
    if not txt:
        sys.exit(f"Config is empty: {path}")
    cfg = yaml.safe_load(txt)
    if cfg is None or not isinstance(cfg, dict):
        sys.exit(f"Config could not be parsed (YAML): {path}")
    return cfg

def run_once(cfg):
    set_seed(cfg.get("seed", 42))
    X, y, task = load_dataset(cfg["dataset"])
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te), _ = make_splits(
        X, y, task,
        test_size=cfg.get("test_size", 0.2),
        val_size=cfg.get("val_size", 0.2),
        seed=cfg.get("seed", 42),
    )
    logs = run_active_learning(
        X_tr, y_tr, X_val, y_val, X_te, y_te,
        task=task,
        hidden_dim=cfg["model"]["hidden_dim"],
        activation=cfg["model"].get("activation", "relu"),
        weight_decay=cfg["train"].get("weight_decay", 1e-4),
        lr=cfg["train"].get("lr", 1e-2),
        batch_size=cfg["train"].get("batch_size", 32),
        patience=cfg["train"].get("early_stopping_patience", 20),
        init_frac=cfg["al"].get("init_frac", 0.1),
        query_size=cfg["al"].get("query_size", 5),
        budget=cfg["al"].get("budget", 200),
        strategy=cfg["al"].get("strategy", "passive"),
        mc_dropout_passes=cfg["al"].get("mc_dropout_passes", 0),
        seed=cfg.get("seed", 42),
    )
    os.makedirs("experiments/results", exist_ok=True)
    out = os.path.join("experiments/results", f"{cfg['name']}.json")
    with open(out, "w") as f:
        json.dump(logs, f, indent=2)
    print(f"Wrote {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", required=True)  # e.g. 0,1,2
    args = ap.parse_args()

    base = load_cfg(args.config)
    seed_list = [int(s) for s in args.seeds.split(",") if s.strip()!=""]

    for s in seed_list:
        cfg = load_cfg(args.config)
        cfg["seed"] = s
        # ensure name exists and suffix it
        if "name" not in cfg or not cfg["name"]:
            cfg["name"] = os.path.splitext(os.path.basename(args.config))[0]
        cfg["name"] = f"{cfg['name']}_s{s}"
        run_once(cfg)

if __name__ == "__main__":
    main()

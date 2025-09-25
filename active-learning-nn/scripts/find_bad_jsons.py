import glob, json, os
bad = []
for p in glob.glob("experiments/results/*.json"):
    try:
        txt = open(p,'r').read().strip()
        if not txt: raise ValueError("empty file")
        data = json.loads(txt)
        if not isinstance(data, list): raise ValueError("not a list of logs")
    except Exception as e:
        bad.append((p, str(e)))
for p, e in bad:
    print(f"BAD: {p} -- {e}")
print(f"Checked {len(glob.glob('experiments/results/*.json'))} files; bad: {len(bad)}")

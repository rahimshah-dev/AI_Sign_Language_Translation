import json
import pickle
from collections import defaultdict

INPUT_PICKLE = "data.pickle"
OUTPUT_JSON = "frontend/model.json"

with open(INPUT_PICKLE, "rb") as f:
    data_dict = pickle.load(f)

data = data_dict.get("data", [])
labels = data_dict.get("labels", [])

if not data:
    raise SystemExit("data.pickle contains no data")

max_len = max(len(x) for x in data)

sums = {}
counts = defaultdict(int)

for vec, label in zip(data, labels):
    if label not in sums:
        sums[label] = [0.0] * max_len
    counts[label] += 1
    padded = list(vec) + [0.0] * (max_len - len(vec))
    for i, v in enumerate(padded):
        sums[label][i] += float(v)


def _label_key(x):
    s = str(x)
    return (0, int(s)) if s.isdigit() else (1, s)


label_ids = sorted(sums.keys(), key=_label_key)
centroids = []
for label in label_ids:
    count = counts[label]
    centroids.append([v / count for v in sums[label]])

label_strs = [str(l) for l in label_ids]
label_map_default = {
    "0": "Yes",
    "1": "No",
    "2": "Hello",
    "3": "I love you",
    "4": "Thank you",
}
label_map = {s: label_map_default.get(s, s) for s in label_strs}

out = {
    "featureLength": max_len,
    "labels": label_strs,
    "labelMap": label_map,
    "centroids": centroids,
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(out, f)

print(f"Wrote {OUTPUT_JSON} with {len(label_strs)} labels and {max_len} features.")

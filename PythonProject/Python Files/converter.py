##Almost Exclusively GPT

import scipy.io
import csv
import json
import re
from collections import defaultdict

# -----------------------------
# Config
# -----------------------------
MAT_FILE = "../../../imdb_crop/imdb.mat"  # or wiki.mat
IMDB_TSV = "../../../name.basics.tsv"
OUTPUT_JSON = "../../../imdbwiki_to_imdb.json"

# -----------------------------
# Utilities
# -----------------------------
def normalize_name(name):
    name = name[0][0].lower()
    name = re.sub(r"[^a-z\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

def matlab_datenum_to_year(d):
    if d <= 0:
        return None
    # MATLAB datenum → year (good enough for matching)
    return int((d - 366) // 365.2425 + 1)

# -----------------------------
# Load IMDB-WIKI
# -----------------------------
print("Loading IMDB-WIKI .mat file...")
mat = scipy.io.loadmat(MAT_FILE)

data = mat['imdb']

names = [n[0] for n in data["name"][0]]
dob = data["dob"][0]
internal_ids = data["celeb_id"][0]

dataset_people = []
for iid, name, d in zip(internal_ids, names, dob):
    dataset_people.append({
        "internal_id": int(iid[0][0]),
        "name": name,
        "birth_year": matlab_datenum_to_year(d[0][0])
    })

print(f"Loaded {len(dataset_people)} identities")

# -----------------------------
# Load IMDb TSV
# -----------------------------
print("Loading IMDb name.basics.tsv...")
imdb_index = defaultdict(list)

with open(IMDB_TSV, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        if row["birthYear"] == "\\N":
            continue

        professions = row["primaryProfession"]
        if "actor" not in professions or "actress" not in professions:
            continue

        imdb_index[normalize_name(row["primaryName"])].append({
            "nconst": row["nconst"],
            "birth_year": int(row["birthYear"])
        })

print(f"Indexed {len(imdb_index)} IMDb names")

# -----------------------------
# Matching
# -----------------------------
print("Matching identities...")
mapping = {}
matched = 0

for p in dataset_people:
    name_key = normalize_name(p["name"])
    birth_year = p["birth_year"]

    if not birth_year or name_key not in imdb_index:
        continue

    candidates = imdb_index[name_key]

    match = None
    for c in candidates:
        if c["birth_year"] == birth_year:
            match = c
            break

    if not match:
        for c in candidates:
            if abs(c["birth_year"] - birth_year) <= 1:
                match = c
                break

    if match:
        mapping[p["internal_id"]] = {
            "name": p["name"],
            "birth_year": birth_year,
            "nconst": match["nconst"],
            "imdb_url": f"https://www.imdb.com/name/{match['nconst']}/"
        }
        matched += 1

print(f"Matched {matched} identities")

# -----------------------------
# Save
# -----------------------------
with open(OUTPUT_JSON, "w") as f:
    json.dump(mapping, f, indent=2)

print(f"Saved mapping to {OUTPUT_JSON}")

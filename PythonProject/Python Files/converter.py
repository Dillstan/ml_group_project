##Almost Exclusively GPT

import scipy.io
import csv
import json
import re
from collections import defaultdict

# -----------------------------
# Config
# -----------------------------
MAT_FILE = "../../../ml_project_files/imdb_crop/imdb.mat"  # or wiki.mat
IMDB_TSV = "../../../ml_project_files/imdb_files/name.basics.tsv"
OUTPUT_JSON = "../../../ml_project_files/imdb_files/imdbwiki_to_imdb.json"

# -----------------------------
# Utilities
# -----------------------------
def normalize_name(name):
    name = name.lower()
    #name = re.sub(r"[^a-z\s]", "", name)
    #name = re.sub(r"\s+", " ", name).strip()
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

# Access the fields correctly
names = data['name'][0, 0]
dob = data['dob'][0, 0]
internal_ids = data['celeb_id'][0, 0]

# Optional: Convert from numpy arrays to lists if needed
names_list = names.flatten().tolist()
dob_list = dob.flatten().tolist()
internal_ids_list = internal_ids.flatten().tolist()

dataset_people = []
for iid, name, d in zip(internal_ids_list, names_list, dob_list):
    dataset_people.append({
        "internal_id": int(iid),
        "name": name[0],
        "birth_year": matlab_datenum_to_year(d)
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

        imdb_index[(row["primaryName"]).lower()].append({
            "name": row["primaryName"],
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

not_added = 0
not_added_names = []

for p in dataset_people:
    name_key = p["name"].lower()
    birth_year = p["birth_year"]

    if name_key not in imdb_index:
        not_added += 1
        not_added_names.append(f"IMDB-WIKI: {name_key}, {birth_year}")
        print("name key not found: "+name_key)
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
            "name": name_key,
            "birth_year": birth_year,
            "nconst": match["nconst"],
            "imdb_url": f"https://www.imdb.com/name/{match['nconst']}/"
        }
        matched += 1
    else:
        not_added += 1
        not_added_names.append(f'IMDB-WIKI: {name_key}, {birth_year} | IMDB: {c["nconst"]}, {c["birth_year"]}')
        print(f'imdb-wiki name: {name_key} ---- imdb name: {c["name"]}')

print(f"Matched {matched} identities")

with open("../../../ml_project_files/imdb_files/json_names_missed.txt", "w") as output:
    distinct = list(dict.fromkeys(not_added_names))
    output.write(str(len(distinct)) + "\n")
    output.write("\n".join(distinct))

# -----------------------------
# Save
# -----------------------------
with open(OUTPUT_JSON, "w") as f:
    json.dump(mapping, f, indent=2)

print(f"Saved mapping to {OUTPUT_JSON}")

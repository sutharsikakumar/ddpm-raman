"""
Download the MLROD Raman CSVs referenced in export_0_3648.csv.

Usage:
    python scripts/get_mlrod_spectra.py \
        --manifest data/mlrod/export_0_3648.csv \
        --dest     data/mlrod/raw_csv
"""
import csv, os, pathlib, requests, concurrent.futures as fut

MLROD_BASE = "https://ahed.nasa.gov/mlrod/v3/files"

def fetch(name, dest):
    url  = f"{MLROD_BASE}/{name}"
    out  = dest / name
    if out.exists():
        return name, "skipped"
    r = requests.get(url, timeout=60)
    if r.status_code == 200:
        out.write_bytes(r.content)
        return name, "ok"
    return name, f"HTTP {r.status_code}"

def main(manifest, dest):
    dest = pathlib.Path(dest); dest.mkdir(parents=True, exist_ok=True)

    wanted = set()
    with open(manifest, newline='') as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            for col in ("TRAINING", "LABELED TEST", "VALIDATION", "TEST"):
                val = row.get(col, "")
                if val:
                    wanted.update(val.split("|"))
                    print(f"Need {len(wanted)} spectra …")
    with fut.ThreadPoolExecutor(max_workers=8) as ex:
        for name, status in ex.map(lambda n: fetch(n, dest), wanted):
            print(f"{status:7} {name}")

if __name__ == "__main__":
    import argparse, textwrap
    p = argparse.ArgumentParser(description="Grab MLROD Raman spectra listed in a manifest")
    p.add_argument("--manifest", required=True)
    p.add_argument("--dest",     required=True)
    args = p.parse_args()
    main(args.manifest, args.dest)

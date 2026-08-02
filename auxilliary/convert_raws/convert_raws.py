from pathlib import Path
import subprocess
from multiprocessing import Pool, cpu_count

# -------------------------
# CONFIG
# -------------------------
ROOT_DIR = Path(r"C:\Users\Admin\Desktop\APP\forwards\data")
EXT = ".arw"


def convert_file(file_path: Path):
    try:
        cmd = [
            "dcraw_emu",
            "-T",
            "-W",
            "-4",
            "-o", "0",
            str(file_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            return f"OK   {file_path.name}"
        else:
            err = result.stderr.strip() or result.stdout.strip()
            return f"FAIL {file_path.name}: rc={result.returncode} {err[:200]}"

    except Exception as e:
        return f"ERROR {file_path.name}: {e}"


def main():
    print(f"Root: {ROOT_DIR}")

    files = list(ROOT_DIR.rglob(f"*{EXT}"))

    print(f"Found {len(files)} RAW files")

    workers = max(1, cpu_count() - 1)
    print(f"Using {workers} workers")

    with Pool(workers) as pool:
        results = pool.map(convert_file, files)

    ok = sum(r.startswith("OK") for r in results)
    skip = sum(r.startswith("SKIP") for r in results)
    fail = sum(r.startswith("FAIL") for r in results)
    err = sum(r.startswith("ERROR") for r in results)

    print("\n--- DONE ---")
    print(f"OK   : {ok}")
    print(f"SKIP : {skip}")
    print(f"FAIL : {fail}")
    print(f"ERROR: {err}")

    for r in results:
        if r.startswith("FAIL") or r.startswith("ERROR"):
            print(r)


if __name__ == "__main__":
    main()
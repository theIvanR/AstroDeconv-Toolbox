from __future__ import annotations

import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import tifffile as tiff


INPUT_DIR = Path(r"C:\Users\Admin\Desktop\APP\forwards\output")
OUTPUT_DIR = Path(r"C:\Users\Admin\Desktop\APP\forwards\temp")

# Set to True if you want every file forced to uint16 for DSS compatibility
FORCE_UINT16 = True


def convert_one(path_in: str, path_out: str, force_uint16: bool = True) -> str:
    in_path = Path(path_in)
    out_path = Path(path_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = tiff.imread(str(in_path))

    if force_uint16 and img.dtype != np.uint16:
        # Clamp negative values and values above 65535 before casting
        if np.issubdtype(img.dtype, np.floating):
            img = np.nan_to_num(img, nan=0.0, posinf=65535.0, neginf=0.0)
        img = np.clip(img, 0, 65535).astype(np.uint16)

    # Write uncompressed, minimal-metadata TIFF
    tiff.imwrite(
        str(out_path),
        img,
        compression=None,
        metadata=None,
    )

    return str(in_path)


def collect_tiffs(root: Path) -> list[tuple[Path, Path]]:
    jobs = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}:
            rel = p.relative_to(root)
            jobs.append((p, OUTPUT_DIR / rel))
    return jobs


def main() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    jobs = collect_tiffs(INPUT_DIR)
    if not jobs:
        print("No TIFF files found.")
        return

    workers = os.cpu_count() or 1
    print(f"Found {len(jobs)} TIFF files.")
    print(f"Using {workers} worker processes.")

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(convert_one, str(src), str(dst), FORCE_UINT16)
            for src, dst in jobs
        ]

        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                src = fut.result()
                print(f"[{done}/{len(jobs)}] OK: {src}")
            except Exception as e:
                print(f"[{done}/{len(jobs)}] FAIL: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
from multiprocessing import Pool, cpu_count
from pathlib import Path
import subprocess


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY_DIR = Path(__file__).resolve().parent

INPUT_DIR = PROJECT_ROOT / "input"
FRAME_TYPES = ("light", "flat", "dark")
EXTENSION = ".arw"

DCRAW = THIRD_PARTY_DIR / "dcraw_emu.exe"
WORKERS = max(1, min(8, cpu_count() - 1))


# ============================================================
# CONVERSION
# ============================================================

def convert_file(file_path: Path) -> str:
    """Write an undemosaiced, linear 16-bit TIFF beside its RAW."""

    out_file = file_path.with_suffix(".tiff")

    cmd = [
        str(DCRAW),
        "-v",
        "-T",
        "-W",
        "-4",
        "-o", "0",
        "-h",                    # Half-size RGB from Bayer data
        "-Z", str(out_file),
        str(file_path),
    ]

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

    except subprocess.CalledProcessError as exc:
        error = (
            exc.stderr.strip()
            or exc.stdout.strip()
            or "Unknown dcraw_emu error"
        )
        raise RuntimeError(
            f"{file_path.name}: conversion failed "
            f"(rc={exc.returncode}): {error[:500]}"
        ) from exc

    except OSError as exc:
        raise RuntimeError(
            f"{file_path.name}: could not execute {DCRAW}: {exc}"
        ) from exc

    if not out_file.is_file():
        raise RuntimeError(
            f"{file_path.name}: expected output was not created: "
            f"{out_file}"
        )

    return f"{file_path.name} -> {out_file.name} ({out_file.stat().st_size:,} bytes)"


# ============================================================
# DISCOVERY
# ============================================================

def build_jobs() -> list[Path]:
    jobs = []

    for frame_type in FRAME_TYPES:
        source_dir = INPUT_DIR / frame_type

        if not source_dir.is_dir():
            print(f"[WARN] Missing directory: {source_dir}")
            continue

        files = sorted(
            p for p in source_dir.rglob("*")
            if p.is_file() and p.suffix.lower() == EXTENSION
        )

        print(f"{frame_type.upper():5}: {len(files)} RAW files")
        jobs.extend(files)

    return jobs


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print(f"Project root: {PROJECT_ROOT}")
    print(f"RAW converter: {DCRAW}")
    print(f"Input: {INPUT_DIR}")

    if not DCRAW.is_file():
        raise FileNotFoundError(f"dcraw_emu not found: {DCRAW}")

    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    jobs = build_jobs()

    print(f"\nTotal RAW files: {len(jobs)}")
    print(f"Workers: {min(WORKERS, len(jobs)) if jobs else 0}")
    print("Output: beside each original RAW file")
    print("Mode: linear 16-bit TIFF, interpolation disabled\n")

    if not jobs:
        print("Nothing to do.")
        return

    with Pool(min(WORKERS, len(jobs))) as pool:
        for result in pool.imap_unordered(convert_file, jobs):
            print(f"[OK] {result}")

    print("\n--- DONE ---")
    print(f"Converted: {len(jobs)}")


if __name__ == "__main__":
    main()

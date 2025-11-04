# src/prepare_data.py
import argparse, shutil
from pathlib import Path

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="flat/recursive images root (e.g., data_raw/train)")
    ap.add_argument("--dst", required=True, help="destination folder with class subfolders (e.g., data)")
    args = ap.parse_args()

    SRC = Path(args.src)
    DST = Path(args.dst)
    CATS = DST / "cats"
    DOGS = DST / "dogs"
    CATS.mkdir(parents=True, exist_ok=True)
    DOGS.mkdir(parents=True, exist_ok=True)

    moved = {"cats": 0, "dogs": 0, "skipped": 0}
    for p in SRC.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext not in VALID_EXT:
            continue

        name = p.name.lower()
        if "cat" in name:
            shutil.copy2(p, CATS / p.name)
            moved["cats"] += 1
        elif "dog" in name:
            shutil.copy2(p, DOGS / p.name)
            moved["dogs"] += 1
        else:
            moved["skipped"] += 1

    print(f"Copied: cats={moved['cats']} dogs={moved['dogs']} skipped={moved['skipped']}")
    print("Destination:", DST.resolve())

if __name__ == "__main__":
    main()

# tools/make_progress_gif.py
import argparse, glob
from pathlib import Path
import imageio.v2 as imageio

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="dir with epoch_XXXX.png")
    ap.add_argument("--out", required=True, help="output gif path")
    ap.add_argument("--fps", type=int, default=3)
    args = ap.parse_args()

    frames = sorted(glob.glob(str(Path(args.samples) / "epoch_*.png")))
    if not frames:
        raise SystemExit("No sample frames found")
    imgs = [imageio.imread(fp) for fp in frames]
    imageio.mimsave(args.out, imgs, fps=args.fps)
    print("Wrote", args.out)

if __name__ == "__main__":
    main()

"""Retired pre-repair traffic implementation. Use the current strict-entry pipeline."""

def main():
    print("This legacy entry point used the withdrawn primitive-action/SF protocol.")
    print("Run: python run.py --help")
    print("Current pipeline: python run.py --stage all --scene-dir ../data/raw_data/manhattan_28x7")
    return 2

if __name__ == "__main__":
    raise SystemExit(main())

"""Back-compat wrapper; the runner now lives in codrawing.cli (codrawing-run)."""

from codrawing.cli import main

if __name__ == "__main__":
    main()

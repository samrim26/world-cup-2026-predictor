"""Enable ``python -m wcp ...``."""
from .app.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

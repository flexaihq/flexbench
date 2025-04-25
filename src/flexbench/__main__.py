import os
from pathlib import Path

from flexbench.main import main

if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent.parent.parent
    os.chdir(root_dir / "src" / "flexbench")
    main()

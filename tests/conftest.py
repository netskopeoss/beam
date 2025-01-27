import sys
from beam.utils import get_project_root

PROJECT_DIR = get_project_root()
root_dir = PROJECT_DIR / "src" / "beam"

sys.path.insert(0, root_dir)
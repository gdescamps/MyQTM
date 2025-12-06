import os

import src


def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(src.__file__)))

import sys
from pathlib import Path


def _add_repo_libs_to_path():
    repo_root = Path(__file__).resolve().parent
    libs_root = repo_root / "libs"
    lib_roots = [
        libs_root / "point-renderer",
        libs_root / "peract_colab",
        libs_root / "YARR",
        libs_root / "RLBench",
        libs_root / "PyRep",
    ]

    for lib_root in lib_roots:
        lib_root_str = str(lib_root)
        if lib_root.exists() and lib_root_str not in sys.path:
            sys.path.insert(0, lib_root_str)


_add_repo_libs_to_path()

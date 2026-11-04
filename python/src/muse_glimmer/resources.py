from pathlib import Path


def resource_path(name: str) -> Path:
    source = Path(__file__).resolve()
    root = source.parents[3]
    if source == root / "python/src/muse_glimmer/resources.py":
        return root / name
    return source.parent / name

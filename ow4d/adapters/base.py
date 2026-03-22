from pathlib import Path

class Adapter:
    name = "base"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self, frames, prompt_groups, width, height):
        raise NotImplementedError

    def info(self):
        return {
            "name": self.name,
            "kwargs": self.kwargs
        }

def ensure_file(path):
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"missing file: {path}")
    return path

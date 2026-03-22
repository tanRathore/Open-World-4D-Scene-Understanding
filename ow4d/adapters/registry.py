from ow4d.adapters.mock import run as run_mock
from ow4d.adapters.json_adapter import JsonAdapter
from ow4d.adapters.grounded_sam2 import GroundedSAM2Adapter

def run_adapter(name, frames, prompt_groups, width, height, **kwargs):
    if name == "mock":
        return run_mock(frames, prompt_groups, width, height)

    if name == "json":
        adapter = JsonAdapter(**kwargs)
        return adapter.run(frames, prompt_groups, width, height)

    if name == "grounded_sam2":
        adapter = GroundedSAM2Adapter(**kwargs)
        return adapter.run(frames, prompt_groups, width, height)

    raise RuntimeError(f"bad adapter: {name}")

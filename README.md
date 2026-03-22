# Toward Open-World 4D Scene Understanding  
### Stabilized Video Tracking with 3D Lifting and Temporal Scene Graphs  
*(ongoing research-engineering project / work in progress)*

This project explores how to move from noisy open-world video predictions toward structured 4D scene understanding.

The main idea is that raw outputs from open-vocabulary grounding and video segmentation models are **not enough on their own** for coherent scene understanding. They often drift over time, vary semantically across frames, and are awkward to use for downstream geometry, graph construction, and future prediction.

This repository builds a controller-side pipeline that takes those raw predictions and turns them into:

- temporally stabilized tracks
- canonicalized object identities
- structured object-centric 4D state
- preliminary 3D lifted state
- temporal scene graphs
- simple future-state / future-graph baseline artifacts

The current project should be understood as **toward** open-world 4D scene understanding, not a finished end-to-end world model. The strongest current contribution is the **controller-side temporal stabilization and structured state pipeline** built on top of foundation-model video outputs.

---

## Demo

**Demo video / artifacts:**  
Add your link here

Example:
`https://drive.google.com/...`

---

## What this project is trying to do

The long-term goal is:

**open-world grounding -> stable tracks -> object-centric structure -> 3D lifting -> temporal scene graphs -> future scene / graph reasoning**

Given a video and natural-language prompts, the full target system should eventually be able to:

- detect arbitrary objects
- segment them
- track them consistently through time
- reduce drift and prompt noise
- lift the result into a more coherent 3D representation
- build temporal scene graphs
- reason about how the scene evolves and what may happen next

Right now, the project already supports several of these stages, but some parts are still preliminary.

---

## Current status

This repository is currently in a **serious research prototype / active development** stage.

### Working now

The current pipeline already supports:

- external GPU inference for open-world grounding / tracking
- normalized GPU-to-controller interchange artifacts
- label canonicalization across prompt variants
- controller-side temporal stabilization
- guarded anchor fusion
- `track_state`
- `object_state_4d`
- `lifted_object_state_4d`
- `scene_graph_4d`
- 2D forecast sample generation and baseline comparison
- 3D forecast sample generation and baseline comparison
- report/demo asset generation under `paper/`

### What is real but still immature

The project now has a real lifted 3D path, including:

- matched depth loading
- intrinsics loading
- box-region depth pooling
- camera-frame backprojection
- `world_xyz`
- `world_velocity_xyz`
- `world_size_wh`
- approximate `bbox_3d`

That said, this is still **preliminary 3D lifting**, not finished world-consistent geometry.

### What is not finished yet

Still under development:

- object-aware / mask-aware depth pooling
- stronger 3D box / extent fitting
- pose-aware world-frame lifting
- stronger multi-object 3D scene graph evaluation
- learned future scene / future graph prediction
- larger-scale benchmark expansion

---

## Architecture

This project is intentionally split into two layers.

### 1. External GPU inference layer

This stage runs the heavy open-world grounding / video tracking stack on Linux GPU infrastructure.

Its job is to produce normalized interchange artifacts:

- `preds.json`
- `anchors.json`

These files are the contract between inference and the controller pipeline.

### 2. Local controller / evaluation layer

This stage lives in `ow4d/` and is where the current core contribution sits.

It handles:

- input inspection
- prompt parsing
- imported prediction loading
- label canonicalization
- temporal stabilization
- anchor fusion
- object-state construction
- preliminary 3D lifting
- temporal scene graph construction
- forecast artifact generation
- local evaluation / summaries / report assets

This split keeps the system modular and makes it easier to improve downstream reasoning without tightly coupling everything to one inference backend.

---

## Canonical pipeline

The current canonical controller path is:

`preds.json -> label canonicalization -> temporal stabilization -> anchor fusion -> track_state -> object_state_4d -> lifted_object_state_4d -> scene_graph_4d`

This is important: the 4D / lifted / graph artifacts are now built from the **same canonical controller-side observation stream**, not from a separate side branch.

---

## Repository layout

Below is the high-level structure of the repository.

```text
Open-World 4D Scene Understanding/
│
├── configs/
│   └── base.yaml
│
├── data/
│   └── raw/
│
├── gpu_runner/
│   ├── README.md
│   ├── configs/
│   ├── scripts/
│   └── nautilus-gs2-pod.yaml
│
├── ow4d/
│   ├── adapters/
│   ├── datasets/
│   ├── stabilization/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── depth_io.py
│   ├── dynamics.py
│   ├── forecast_3d.py
│   ├── forecast_baseline.py
│   ├── forecast_prep.py
│   ├── future_graph_baseline.py
│   ├── future_graph_prep.py
│   ├── fusion.py
│   ├── inputs.py
│   ├── io.py
│   ├── lifters.py
│   ├── lifting.py
│   ├── manifest.py
│   ├── metrics.py
│   ├── object_graph.py
│   ├── pipeline.py
│   ├── prompts.py
│   ├── render.py
│   ├── scene_graph.py
│   ├── scene_graph_4d.py
│   ├── track_state.py
│   ├── tracks.py
│   └── video.py
│
├── outputs/
│
├── paper/
│   ├── figures/
│   ├── tables/
│   └── tools/
│
├── scripts/
│   ├── discover_davis_jobs.py
│   ├── run_davis_benchmark.py
│   ├── prepare_davis_nautilus_batch.py
│   └── compatibility / wrapper utilities
│
├── tmp/
│
└── README.md
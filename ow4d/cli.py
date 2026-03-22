import argparse
from pprint import pprint

from ow4d.config import load_config
from ow4d.prompts import parse_prompt_groups
from ow4d.inputs import inspect_input
from ow4d.pipeline import run_core
from ow4d.datasets.davis import list_davis_sequences

def cmd_show_config(args):
    cfg = load_config(args.config)
    pprint(cfg)

def cmd_scan_input(args):
    info = inspect_input(args.input)
    pprint(info)

def cmd_plan_prompts(args):
    groups = parse_prompt_groups(args.text)
    pprint(groups)

def cmd_list_davis(args):
    rows = list_davis_sequences(args.root)
    for row in rows:
        print(f'{row["name"]}\t{row["frames"]}\t{row["path"]}')

def cmd_run_core(args):
    res = run_core(
        input_path=args.input,
        prompt_text=args.text,
        config_path=args.config,
        out_dir=args.out,
        adapter=args.adapter,
        render=not args.no_render,
        stride=args.stride,
        max_frames=args.max_frames,
        pred_path=args.pred_path,
        anchor_path=args.anchor_path,
        lift_mode=args.lift_mode,
        depth_dir=args.depth_dir,
        intrinsics_path=args.intrinsics_path,
        temporal_stabilization=(False if args.no_temporal_stabilization else None),
        anchor_fusion=(False if args.no_anchor_fusion else None),
        stabilization_alpha=args.stabilization_alpha,
        stabilization_max_gap=args.stabilization_max_gap,
        stabilization_max_center_step=args.stabilization_max_center_step,
        anchor_fusion_window=args.anchor_fusion_window,
        anchor_fusion_weight=args.anchor_fusion_weight,
    )
    pprint(res)
    print("done")

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("show-config")
    a.add_argument("--config", default="configs/base.yaml")
    a.set_defaults(func=cmd_show_config)

    b = sub.add_parser("scan-input")
    b.add_argument("--input", required=True)
    b.set_defaults(func=cmd_scan_input)

    c = sub.add_parser("plan-prompts")
    c.add_argument("--text", required=True)
    c.set_defaults(func=cmd_plan_prompts)

    d = sub.add_parser("list-davis")
    d.add_argument("--root", required=True)
    d.set_defaults(func=cmd_list_davis)

    e = sub.add_parser("run-core")
    e.add_argument("--input", required=True)
    e.add_argument("--text", required=True)
    e.add_argument("--config", default="configs/base.yaml")
    e.add_argument("--out", default="outputs/run1")
    e.add_argument("--adapter", default=None)
    e.add_argument("--stride", type=int, default=None)
    e.add_argument("--max-frames", type=int, default=None)
    e.add_argument("--pred-path", default=None)
    e.add_argument("--anchor-path", default=None)
    e.add_argument("--lift-mode", default=None)
    e.add_argument("--depth-dir", default=None)
    e.add_argument("--intrinsics-path", default=None)
    e.add_argument("--no-temporal-stabilization", action="store_true")
    e.add_argument("--no-anchor-fusion", action="store_true")
    e.add_argument("--stabilization-alpha", type=float, default=None)
    e.add_argument("--stabilization-max-gap", type=int, default=None)
    e.add_argument("--stabilization-max-center-step", type=float, default=None)
    e.add_argument("--anchor-fusion-window", type=int, default=None)
    e.add_argument("--anchor-fusion-weight", type=float, default=None)
    e.add_argument("--no-render", action="store_true")
    e.set_defaults(func=cmd_run_core)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()

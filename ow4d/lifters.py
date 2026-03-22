from copy import deepcopy

from ow4d.depth_io import (
    backproject_xy_depth,
    build_depth_match_table,
    sample_depth_at_xy,
    sample_depth_in_box,
)


def _base_lift_meta(mode, depth_dir=None, intrinsics_path=None, notes=""):
    return {
        "mode": mode,
        "used_depth": False,
        "used_intrinsics": False,
        "depth_dir": depth_dir,
        "intrinsics_path": intrinsics_path,
        "has_depth_dir": bool(depth_dir),
        "has_intrinsics_path": bool(intrinsics_path),
        "ready_for_true_3d": bool(depth_dir) and bool(intrinsics_path),
        "notes": notes,
    }


def _image_wh_to_world_wh(image_w, image_h, depth, intrinsics):
    if depth is None or intrinsics is None:
        return None
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    world_w = float(image_w) * float(depth) / fx
    world_h = float(image_h) * float(depth) / fy
    return [round(world_w, 6), round(world_h, 6)]


def _box_xyxy_from_center(cx, cy, image_w, image_h):
    half_w = float(image_w) / 2.0
    half_h = float(image_h) / 2.0
    return [
        float(cx) - half_w,
        float(cy) - half_h,
        float(cx) + half_w,
        float(cy) + half_h,
    ]


def _depth_from_box_or_center(depth_path, cx, cy, image_w, image_h):
    box_xyxy = _box_xyxy_from_center(cx, cy, image_w, image_h)
    depth, stats = sample_depth_in_box(depth_path, *box_xyxy, return_stats=True)

    if depth is not None:
        stats["mode"] = "box_pool"
        return depth, stats

    depth = sample_depth_at_xy(depth_path, cx, cy)
    if depth is None:
        return None, {
            "mode": "missing",
            "box_xyxy": [round(v, 3) for v in box_xyxy],
            "value_count": 0,
            "median": None,
            "depth_span": None,
        }

    return depth, {
        "mode": "center_patch",
        "box_xyxy": [round(v, 3) for v in box_xyxy],
        "value_count": 1,
        "median": round(float(depth), 6),
        "depth_span": 0.0,
    }


def _clone_image_plane_object_state(object_state_4d, depth_dir=None, intrinsics_path=None):
    out = deepcopy(object_state_4d)
    out["representation"] = "lifted_object_state_4d"
    out["lift_mode"] = "image_plane"
    out["valid_3d"] = False
    out["coord_frame"] = "image_plane"
    out["depth_match"] = build_depth_match_table(
        object_state_4d,
        depth_dir=depth_dir,
        intrinsics_path=intrinsics_path,
    )
    meta = _base_lift_meta(
        mode="image_plane",
        depth_dir=depth_dir,
        intrinsics_path=intrinsics_path,
        notes="pass-through lift; preserves 2D/image-plane state as the 4D object contract",
    )
    meta["matched_state_count"] = out["depth_match"]["matched_state_count"]
    meta["depth_file_count"] = out["depth_match"]["depth_file_count"]
    meta["used_intrinsics"] = out["depth_match"]["intrinsics"] is not None
    out["lift_meta"] = meta
    return out


def _stub_3d_object_state(object_state_4d, depth_dir=None, intrinsics_path=None):
    out = deepcopy(object_state_4d)
    out["representation"] = "lifted_object_state_4d"
    out["lift_mode"] = "stub_3d"
    out["valid_3d"] = False
    out["coord_frame"] = "world_stub"
    out["depth_match"] = build_depth_match_table(
        object_state_4d,
        depth_dir=depth_dir,
        intrinsics_path=intrinsics_path,
    )

    for obj in out.get("objects", []):
        for s in obj.get("states", []):
            cx, cy = s["image_center_xy_norm"]
            s["depth"] = None
            s["world_xyz"] = [cx, cy, None]
            s["world_velocity_xyz"] = [None, None, None]
            s["bbox_3d"] = None
            s["world_size_wh"] = None
            s["world_size_velocity_wh"] = None
            s["valid_3d"] = False
            s["coord_frame"] = "world_stub"

    meta = _base_lift_meta(
        mode="stub_3d",
        depth_dir=depth_dir,
        intrinsics_path=intrinsics_path,
        notes="placeholder world-state scaffold for future true 3D lifting",
    )
    meta["matched_state_count"] = out["depth_match"]["matched_state_count"]
    meta["depth_file_count"] = out["depth_match"]["depth_file_count"]
    meta["used_intrinsics"] = out["depth_match"]["intrinsics"] is not None
    out["lift_meta"] = meta
    return out


def _center_depth_3d_object_state(object_state_4d, depth_dir=None, intrinsics_path=None):
    out = deepcopy(object_state_4d)
    out["representation"] = "lifted_object_state_4d"
    out["lift_mode"] = "center_depth_3d"
    out["valid_3d"] = False
    out["coord_frame"] = "camera"
    out["depth_match"] = build_depth_match_table(
        object_state_4d,
        depth_dir=depth_dir,
        intrinsics_path=intrinsics_path,
    )

    intrinsics = out["depth_match"]["intrinsics"]
    depth_by_frame = {}
    for row in out["depth_match"]["rows"]:
        if row["depth_path"] and row["frame_idx"] not in depth_by_frame:
            depth_by_frame[row["frame_idx"]] = row["depth_path"]

    valid_count = 0

    for obj in out.get("objects", []):
        prev_world = None
        prev_world_size = None
        prev_frame = None

        for s in obj.get("states", []):
            frame_idx = int(s["frame_idx"])
            cx, cy = s["image_center_xy"]
            image_w, image_h = s["image_size_wh"]
            depth_path = depth_by_frame.get(frame_idx)

            depth = None
            depth_stats = None
            world = None
            world_size = None

            if depth_path and intrinsics is not None:
                depth, depth_stats = _depth_from_box_or_center(
                    depth_path,
                    cx,
                    cy,
                    image_w,
                    image_h,
                )
                if depth is not None:
                    world = backproject_xy_depth(cx, cy, depth, intrinsics)
                    world_size = _image_wh_to_world_wh(image_w, image_h, depth, intrinsics)

            s["depth"] = depth
            s["depth_stats"] = depth_stats
            s["world_xyz"] = world
            s["world_size_wh"] = world_size
            s["coord_frame"] = "camera"

            if world is not None and world_size is not None:
                z_size = 0.0
                if depth_stats is not None and depth_stats.get("depth_span") is not None:
                    z_size = round(float(depth_stats["depth_span"]), 6)

                s["bbox_3d"] = {
                    "center_xyz": world,
                    "size_xyz": [world_size[0], world_size[1], z_size],
                    "mode": "box_pool_depth_3d",
                }
                s["valid_3d"] = True
                valid_count += 1

                if prev_world is not None and prev_frame is not None and frame_idx != prev_frame:
                    dt = frame_idx - prev_frame
                    s["world_velocity_xyz"] = [
                        round((world[0] - prev_world[0]) / dt, 6),
                        round((world[1] - prev_world[1]) / dt, 6),
                        round((world[2] - prev_world[2]) / dt, 6),
                    ]
                else:
                    s["world_velocity_xyz"] = [0.0, 0.0, 0.0]

                if prev_world_size is not None and prev_frame is not None and frame_idx != prev_frame:
                    dt = frame_idx - prev_frame
                    s["world_size_velocity_wh"] = [
                        round((world_size[0] - prev_world_size[0]) / dt, 6),
                        round((world_size[1] - prev_world_size[1]) / dt, 6),
                    ]
                else:
                    s["world_size_velocity_wh"] = [0.0, 0.0]

                prev_world = world
                prev_world_size = world_size
                prev_frame = frame_idx
            else:
                s["bbox_3d"] = None
                s["valid_3d"] = False
                s["world_velocity_xyz"] = None
                s["world_size_velocity_wh"] = None

    out["valid_3d"] = valid_count > 0
    meta = _base_lift_meta(
        mode="center_depth_3d",
        depth_dir=depth_dir,
        intrinsics_path=intrinsics_path,
        notes="box-region depth pooling with center fallback, then pinhole backprojection using matched depth frames and camera intrinsics",
    )
    meta["matched_state_count"] = out["depth_match"]["matched_state_count"]
    meta["depth_file_count"] = out["depth_match"]["depth_file_count"]
    meta["used_depth"] = valid_count > 0
    meta["used_intrinsics"] = intrinsics is not None
    meta["valid_3d_state_count"] = valid_count
    out["lift_meta"] = meta
    return out


def lift_object_state_4d(
    object_state_4d,
    video_info,
    lift_mode="image_plane",
    depth_dir=None,
    intrinsics_path=None,
    **kwargs,
):
    lift_mode = str(lift_mode or "image_plane").strip().lower()

    if lift_mode == "image_plane":
        return _clone_image_plane_object_state(
            object_state_4d,
            depth_dir=depth_dir,
            intrinsics_path=intrinsics_path,
        )

    if lift_mode == "stub_3d":
        return _stub_3d_object_state(
            object_state_4d,
            depth_dir=depth_dir,
            intrinsics_path=intrinsics_path,
        )

    if lift_mode == "center_depth_3d":
        return _center_depth_3d_object_state(
            object_state_4d,
            depth_dir=depth_dir,
            intrinsics_path=intrinsics_path,
        )

    raise ValueError(f"unsupported lift_mode: {lift_mode}")

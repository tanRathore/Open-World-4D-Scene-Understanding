def build_object_state_4d(track_state, video_info):
    width = max(int(video_info.get("width", 0)), 1)
    height = max(int(video_info.get("height", 0)), 1)

    objects = []
    for track in track_state.get("tracks", []):
        states = []
        for s in track.get("states", []):
            cx, cy = s["center"]
            w, h = s["size"]
            vx, vy = s["velocity"]

            states.append({
                "frame_idx": int(s["frame_idx"]),
                "image_center_xy": [cx, cy],
                "image_center_xy_norm": [round(cx / width, 6), round(cy / height, 6)],
                "image_size_wh": [w, h],
                "image_size_wh_norm": [round(w / width, 6), round(h / height, 6)],
                "velocity_xy": [vx, vy],
                "speed_2d": float(s["speed"]),
                "depth": None,
                "world_xyz": None,
                "world_velocity_xyz": None,
                "bbox_3d": None,
                "valid_3d": False,
                "coord_frame": "image_plane",
                "gap_filled": bool(s.get("gap_filled", False)),
                "score": float(s.get("score", 0.0)),
            })

        objects.append({
            "track_id": track["track_id"],
            "label": track["label"],
            "start_frame": track["start_frame"],
            "end_frame": track["end_frame"],
            "num_frames": track["num_frames"],
            "mean_speed_2d": track["mean_speed"],
            "max_speed_2d": track["max_speed"],
            "states": states,
        })

    return {
        "representation": "object_state_4d",
        "coord_frame": "image_plane",
        "valid_3d": False,
        "track_count": len(objects),
        "objects": objects,
    }

import math


def _valid_state(s):
    world = s.get("world_xyz")
    if not s.get("valid_3d", False):
        return False
    if not isinstance(world, list) or len(world) != 3:
        return False
    return all(v is not None for v in world)


def _world_l2(a, b):
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    dz = float(a[2]) - float(b[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def build_forecast_samples_3d(lifted_object_state_4d, obs_len=4, pred_len=2):
    samples = []

    for obj in lifted_object_state_4d.get("objects", []):
        states = obj.get("states", [])
        if len(states) < obs_len + pred_len:
            continue

        for start in range(0, len(states) - obs_len - pred_len + 1):
            obs = states[start:start + obs_len]
            fut = states[start + obs_len:start + obs_len + pred_len]

            window = obs + fut
            ok = True
            for i in range(1, len(window)):
                if int(window[i]["frame_idx"]) != int(window[i - 1]["frame_idx"]) + 1:
                    ok = False
                    break
            if not ok or not all(_valid_state(s) for s in window):
                continue

            samples.append({
                "track_id": obj["track_id"],
                "label": obj["label"],
                "obs_start_frame": int(obs[0]["frame_idx"]),
                "obs_end_frame": int(obs[-1]["frame_idx"]),
                "pred_end_frame": int(fut[-1]["frame_idx"]),
                "obs_len": obs_len,
                "pred_len": pred_len,
                "obs": [
                    {
                        "frame_idx": int(s["frame_idx"]),
                        "world_xyz": s["world_xyz"],
                        "world_velocity_xyz": s["world_velocity_xyz"],
                        "depth": s["depth"],
                    }
                    for s in obs
                ],
                "target": [
                    {
                        "frame_idx": int(s["frame_idx"]),
                        "world_xyz": s["world_xyz"],
                        "world_velocity_xyz": s["world_velocity_xyz"],
                        "depth": s["depth"],
                    }
                    for s in fut
                ],
            })

    return {
        "sample_count": len(samples),
        "obs_len": obs_len,
        "pred_len": pred_len,
        "samples": samples,
    }


def _eval(preds):
    errs = []
    for item in preds:
        for p, y in zip(item["prediction"], item["target"]):
            errs.append(_world_l2(p["world_xyz"], y["world_xyz"]))

    return {
        "sample_count": len(preds),
        "mean_world_l2": round(sum(errs) / max(len(errs), 1), 6),
        "max_world_l2": round(max(errs), 6) if errs else 0.0,
    }


def build_constant_position_baseline_3d(forecast_samples_3d):
    preds = []

    for sample in forecast_samples_3d.get("samples", []):
        obs = sample["obs"]
        tgt = sample["target"]
        last = obs[-1]

        pred = []
        for step in range(1, len(tgt) + 1):
            pred.append({
                "step": step,
                "world_xyz": list(last["world_xyz"]),
            })

        preds.append({
            "track_id": sample["track_id"],
            "label": sample["label"],
            "obs_start_frame": sample["obs_start_frame"],
            "obs_end_frame": sample["obs_end_frame"],
            "pred_end_frame": sample["pred_end_frame"],
            "prediction": pred,
            "target": tgt,
        })

    return {
        "baseline": "constant_position_3d",
        "metrics": _eval(preds),
        "predictions": preds,
    }


def build_constant_velocity_baseline_3d(forecast_samples_3d):
    preds = []

    for sample in forecast_samples_3d.get("samples", []):
        obs = sample["obs"]
        tgt = sample["target"]
        last = obs[-1]
        last2 = obs[-2] if len(obs) >= 2 else obs[-1]

        vel = last.get("world_velocity_xyz")
        if not isinstance(vel, list) or len(vel) != 3 or any(v is None for v in vel):
            vel = [
                float(last["world_xyz"][0]) - float(last2["world_xyz"][0]),
                float(last["world_xyz"][1]) - float(last2["world_xyz"][1]),
                float(last["world_xyz"][2]) - float(last2["world_xyz"][2]),
            ]

        cur = list(last["world_xyz"])
        pred = []
        for step in range(1, len(tgt) + 1):
            cur = [
                round(float(cur[0]) + float(vel[0]), 6),
                round(float(cur[1]) + float(vel[1]), 6),
                round(float(cur[2]) + float(vel[2]), 6),
            ]
            pred.append({
                "step": step,
                "world_xyz": list(cur),
            })

        preds.append({
            "track_id": sample["track_id"],
            "label": sample["label"],
            "obs_start_frame": sample["obs_start_frame"],
            "obs_end_frame": sample["obs_end_frame"],
            "pred_end_frame": sample["pred_end_frame"],
            "prediction": pred,
            "target": tgt,
        })

    return {
        "baseline": "constant_velocity_3d",
        "metrics": _eval(preds),
        "predictions": preds,
    }


def build_forecast_baseline_compare_3d(cp3d, cv3d):
    cp = cp3d["metrics"]
    cv = cv3d["metrics"]

    winner_by_metric = {
        "mean_world_l2": "constant_velocity_3d" if cv["mean_world_l2"] <= cp["mean_world_l2"] else "constant_position_3d",
        "max_world_l2": "constant_velocity_3d" if cv["max_world_l2"] <= cp["max_world_l2"] else "constant_position_3d",
    }

    return {
        "sample_count": cv["sample_count"],
        "baselines": {
            "constant_position_3d": cp,
            "constant_velocity_3d": cv,
        },
        "winner_by_metric": winner_by_metric,
        "best_mean_world_l2_model": winner_by_metric["mean_world_l2"],
        "best_max_world_l2_model": winner_by_metric["max_world_l2"],
        "delta_cv_minus_cp": {
            "mean_world_l2": round(cv["mean_world_l2"] - cp["mean_world_l2"], 6),
            "max_world_l2": round(cv["max_world_l2"] - cp["max_world_l2"], 6),
        },
    }

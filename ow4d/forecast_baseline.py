import math


def _roll_constant_position(last, pred_len):
    cur_center = list(last["center"])
    cur_size = list(last["size"])
    out = []
    for step in range(1, pred_len + 1):
        out.append({
            "step": step,
            "center": cur_center[:],
            "size": cur_size[:],
            "velocity": [0.0, 0.0],
            "speed": 0.0,
        })
    return out


def _roll_constant_velocity(last, pred_len):
    cur_center = list(last["center"])
    cur_size = list(last["size"])
    vel = list(last["velocity"])
    size_vel = list(last.get("size_velocity", [0, 0]))

    out = []
    for step in range(1, pred_len + 1):
        cur_center = [
            round(cur_center[0] + vel[0], 3),
            round(cur_center[1] + vel[1], 3),
        ]
        cur_size = [
            int(round(cur_size[0] + size_vel[0])),
            int(round(cur_size[1] + size_vel[1])),
        ]
        speed = round(math.sqrt(vel[0] ** 2 + vel[1] ** 2), 3)
        out.append({
            "step": step,
            "center": cur_center[:],
            "size": cur_size[:],
            "velocity": vel[:],
            "speed": speed,
        })
    return out


def _roll_hybrid_velocity_position(last, pred_len):
    cur_center = list(last["center"])
    cur_size = list(last["size"])
    vel = list(last["velocity"])

    out = []
    for step in range(1, pred_len + 1):
        cur_center = [
            round(cur_center[0] + vel[0], 3),
            round(cur_center[1] + vel[1], 3),
        ]
        speed = round(math.sqrt(vel[0] ** 2 + vel[1] ** 2), 3)
        out.append({
            "step": step,
            "center": cur_center[:],
            "size": cur_size[:],
            "velocity": vel[:],
            "speed": speed,
        })
    return out


def _center_err(a, b):
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return math.sqrt(dx * dx + dy * dy)


def _size_err(a, b):
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _evaluate_predictions(preds):
    center_errs = []
    size_errs = []

    for item in preds:
        for p, y in zip(item["prediction"], item["target"]):
            center_errs.append(_center_err(p["center"], y["center"]))
            size_errs.append(_size_err(p["size"], y["size"]))

    return {
        "sample_count": len(preds),
        "mean_center_l2": round(sum(center_errs) / max(len(center_errs), 1), 4),
        "max_center_l2": round(max(center_errs), 4) if center_errs else 0.0,
        "mean_size_l1": round(sum(size_errs) / max(len(size_errs), 1), 4),
        "max_size_l1": round(max(size_errs), 4) if size_errs else 0.0,
    }


def build_constant_position_baseline(forecast_samples):
    samples = forecast_samples.get("samples", [])
    preds = []

    for sample in samples:
        obs = sample["obs"]
        tgt = sample["target"]
        if not obs or not tgt:
            continue

        last = obs[-1]
        pred = _roll_constant_position(last, len(tgt))

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
        "baseline": "constant_position",
        "metrics": _evaluate_predictions(preds),
        "predictions": preds,
    }


def build_constant_velocity_baseline(forecast_samples):
    samples = forecast_samples.get("samples", [])
    preds = []

    for sample in samples:
        obs = sample["obs"]
        tgt = sample["target"]
        if not obs or not tgt:
            continue

        last = obs[-1]
        last2 = obs[-2] if len(obs) >= 2 else obs[-1]

        last_state = {
            "center": last["center"],
            "size": last["size"],
            "velocity": last["velocity"],
            "size_velocity": [
                int(last["size"][0] - last2["size"][0]),
                int(last["size"][1] - last2["size"][1]),
            ],
        }

        pred = _roll_constant_velocity(last_state, len(tgt))

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
        "baseline": "constant_velocity",
        "metrics": _evaluate_predictions(preds),
        "predictions": preds,
    }


def build_hybrid_velocity_position_baseline(forecast_samples):
    samples = forecast_samples.get("samples", [])
    preds = []

    for sample in samples:
        obs = sample["obs"]
        tgt = sample["target"]
        if not obs or not tgt:
            continue

        last = obs[-1]
        last_state = {
            "center": last["center"],
            "size": last["size"],
            "velocity": last["velocity"],
        }

        pred = _roll_hybrid_velocity_position(last_state, len(tgt))

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
        "baseline": "hybrid_velocity_position",
        "metrics": _evaluate_predictions(preds),
        "predictions": preds,
    }


def build_forecast_baseline_compare(forecast_baseline_cp, forecast_baseline_cv, forecast_baseline_hybrid):
    cp = forecast_baseline_cp["metrics"]
    cv = forecast_baseline_cv["metrics"]
    hy = forecast_baseline_hybrid["metrics"]

    metric_names = ["mean_center_l2", "max_center_l2", "mean_size_l1", "max_size_l1"]
    metric_to_vals = {
        "constant_position": cp,
        "constant_velocity": cv,
        "hybrid_velocity_position": hy,
    }

    winner_by_metric = {}
    for metric in metric_names:
        winner_by_metric[metric] = min(
            metric_to_vals.keys(),
            key=lambda name: metric_to_vals[name][metric],
        )

    return {
        "sample_count": cv["sample_count"],
        "baselines": {
            "constant_position": cp,
            "constant_velocity": cv,
            "hybrid_velocity_position": hy,
        },
        "winner_by_metric": winner_by_metric,
        "best_center_model": winner_by_metric["mean_center_l2"],
        "best_size_model": winner_by_metric["mean_size_l1"],
        "delta_hybrid_minus_cp": {
            "mean_center_l2": round(hy["mean_center_l2"] - cp["mean_center_l2"], 4),
            "max_center_l2": round(hy["max_center_l2"] - cp["max_center_l2"], 4),
            "mean_size_l1": round(hy["mean_size_l1"] - cp["mean_size_l1"], 4),
            "max_size_l1": round(hy["max_size_l1"] - cp["max_size_l1"], 4),
        },
        "delta_hybrid_minus_cv": {
            "mean_center_l2": round(hy["mean_center_l2"] - cv["mean_center_l2"], 4),
            "max_center_l2": round(hy["max_center_l2"] - cv["max_center_l2"], 4),
            "mean_size_l1": round(hy["mean_size_l1"] - cv["mean_size_l1"], 4),
            "max_size_l1": round(hy["max_size_l1"] - cv["max_size_l1"], 4),
        },
    }

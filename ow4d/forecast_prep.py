def build_forecast_samples(track_state, obs_len=4, pred_len=2, min_speed=0.0):
    tracks = track_state.get("tracks", [])
    samples = []

    for track in tracks:
        states = track.get("states", [])
        if len(states) < obs_len + pred_len:
            continue

        for start in range(0, len(states) - obs_len - pred_len + 1):
            obs = states[start:start + obs_len]
            fut = states[start + obs_len:start + obs_len + pred_len]

            mean_obs_speed = sum(float(s.get("speed", 0.0)) for s in obs) / max(len(obs), 1)
            if mean_obs_speed < min_speed:
                continue

            sample = {
                "track_id": track["track_id"],
                "label": track["label"],
                "obs_start_frame": int(obs[0]["frame_idx"]),
                "obs_end_frame": int(obs[-1]["frame_idx"]),
                "pred_end_frame": int(fut[-1]["frame_idx"]),
                "obs_len": obs_len,
                "pred_len": pred_len,
                "obs": [
                    {
                        "frame_idx": int(s["frame_idx"]),
                        "center": s["center"],
                        "size": s["size"],
                        "velocity": s["velocity"],
                        "speed": s["speed"],
                    }
                    for s in obs
                ],
                "target": [
                    {
                        "frame_idx": int(s["frame_idx"]),
                        "center": s["center"],
                        "size": s["size"],
                        "velocity": s["velocity"],
                        "speed": s["speed"],
                    }
                    for s in fut
                ],
            }
            samples.append(sample)

    return {
        "sample_count": len(samples),
        "obs_len": obs_len,
        "pred_len": pred_len,
        "samples": samples,
    }

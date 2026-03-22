def build_future_graph_samples_from_scene_graph_4d(scene_graph_4d, obs_len=4, pred_len=2):
    frames = scene_graph_4d.get("frames", [])
    samples = []

    need = obs_len + pred_len
    for start in range(0, len(frames) - need + 1):
        window = frames[start:start + need]

        contiguous = True
        for i in range(1, len(window)):
            if int(window[i]["frame_idx"]) != int(window[i - 1]["frame_idx"]) + 1:
                contiguous = False
                break
        if not contiguous:
            continue

        obs_frames = window[:obs_len]
        tgt_frames = window[obs_len:]

        obs_graphs = []
        tgt_graphs = []

        for item in obs_frames:
            obs_graphs.append({
                "frame_idx": int(item["frame_idx"]),
                "nodes": item.get("nodes", []),
                "relations": item.get("spatial_edges", []),
            })

        for item in tgt_frames:
            tgt_graphs.append({
                "frame_idx": int(item["frame_idx"]),
                "nodes": item.get("nodes", []),
                "relations": item.get("spatial_edges", []),
            })

        samples.append({
            "obs_start_frame": int(obs_frames[0]["frame_idx"]),
            "obs_end_frame": int(obs_frames[-1]["frame_idx"]),
            "pred_end_frame": int(tgt_frames[-1]["frame_idx"]),
            "obs_len": obs_len,
            "pred_len": pred_len,
            "obs_graphs": obs_graphs,
            "target_graphs": tgt_graphs,
        })

    return {
        "sample_count": len(samples),
        "obs_len": obs_len,
        "pred_len": pred_len,
        "samples": samples,
    }

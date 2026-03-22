from datetime import datetime, timezone

def make_manifest(video_path, prompt_text, prompt_groups, cfg, info, frames, observations, out_dir):
    return {
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "video_path": str(video_path),
        "prompt_text": prompt_text,
        "prompt_groups": prompt_groups,
        "config": cfg,
        "video_info": info,
        "frame_count": len(frames),
        "obs_count": len(observations),
        "track_count": len(sorted({x["track_id"] for x in observations})),
        "out_dir": str(out_dir),
        "files": {
            "video_info": "video_info.json",
            "prompts": "prompts.json",
            "frames": "frames.json",
            "tracks": "tracks.json",
            "scene_graph": "scene_graph.json",
            "track_summary": "track_summary.csv",
            "manifest": "manifest.json",
            "preview": "preview_keyframes.mp4"
        }
    }

import re

def _slug(text):
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "item"

def parse_prompt_groups(text):
    groups = []
    if not text.strip():
        return groups

    for chunk in text.split("|"):
        chunk = chunk.strip()
        if not chunk:
            continue

        if "=" in chunk:
            name, raw_prompts = chunk.split("=", 1)
            name = _slug(name)
            prompts = [p.strip() for p in raw_prompts.split(",") if p.strip()]
        else:
            prompts = [p.strip() for p in chunk.split(",") if p.strip()]
            name = _slug(prompts[0]) if prompts else "item"

        if prompts:
            groups.append({
                "name": name,
                "prompts": prompts
            })

    return groups

def flat_prompt_list(groups):
    out = []
    for g in groups:
        out.extend(g["prompts"])
    return out

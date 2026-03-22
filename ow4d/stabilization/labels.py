import re


def normalize_label_text(text):
    text = str(text or "").strip().lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _token_key(text):
    toks = normalize_label_text(text).split()
    toks = [t for t in toks if t]
    return " ".join(sorted(set(toks)))


def build_prompt_alias_index(prompt_groups):
    alias_to_name = {}
    token_to_name = {}

    for group in prompt_groups:
        name = str(group.get("name", "")).strip()
        if not name:
            continue

        candidates = [name]
        candidates.extend(group.get("prompts", []) or [])

        for raw in candidates:
            norm = normalize_label_text(raw)
            if norm and norm not in alias_to_name:
                alias_to_name[norm] = name

            tok = _token_key(raw)
            if tok and tok not in token_to_name:
                token_to_name[tok] = name

    return {
        "alias_to_name": alias_to_name,
        "token_to_name": token_to_name,
    }


def canonical_label_for_text(text, alias_index):
    raw = str(text or "")
    norm = normalize_label_text(raw)
    if not norm:
        return raw

    alias_to_name = alias_index.get("alias_to_name", {})
    token_to_name = alias_index.get("token_to_name", {})

    if norm in alias_to_name:
        return alias_to_name[norm]

    tok = _token_key(norm)
    if tok in token_to_name:
        return token_to_name[tok]

    raw_tokens = set(norm.split())
    if raw_tokens:
        best_name = None
        best_score = 0.0
        for alias, name in alias_to_name.items():
            alias_tokens = set(alias.split())
            if not alias_tokens:
                continue
            inter = len(raw_tokens & alias_tokens)
            if inter == 0:
                continue
            score = inter / max(len(raw_tokens), len(alias_tokens))
            if score > best_score:
                best_name = name
                best_score = score
        if best_name is not None and best_score >= 0.5:
            return best_name

    return raw


def canonicalize_rows(rows, prompt_groups, label_key="label", out_key=None):
    out_key = out_key or label_key
    alias_index = build_prompt_alias_index(prompt_groups)

    out = []
    changed = 0

    for row in rows:
        row2 = dict(row)
        raw = row2.get(label_key, "")
        canon = canonical_label_for_text(raw, alias_index)

        row2[out_key] = canon
        row2["raw_label"] = raw
        row2["label_canonicalized"] = canon != raw
        if canon != raw:
            changed += 1

        out.append(row2)

    return out, {
        "rows": len(out),
        "changed": changed,
    }

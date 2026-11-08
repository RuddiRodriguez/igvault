from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .llm import LLM
from .metadata import load_records
from .utils import (
    banner,
    folder_name,
    iso_now,
    log,
    one_line,
    transcript_path,
    warn,
    write_json,
)

GENERIC_HASHTAGS = {
    "reels", "reel", "viral", "fyp", "foryou", "foryoupage", "explore",
    "instagram", "instagood", "trending", "trend", "love", "follow",
    "explorepage", "viralreels", "reelsinstagram", "reelsvideo",
}

NAMING_SYSTEM = (
    "You organize a personal library of saved Instagram videos into folders. "
    "You read captions and transcripts and produce short, clear, folder-safe topic names."
)


# --------------------------------------------------------------------------- #
# text / vectors
# --------------------------------------------------------------------------- #
def _load_transcript(cfg, record: dict[str, Any]) -> str:
    path = transcript_path(cfg, record["shortcode"])
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _document(record: dict[str, Any], transcript: str) -> str:
    parts: list[str] = []
    caption = (record.get("caption") or "").strip()
    if caption:
        parts.append(caption)
    hashtags = record.get("hashtags") or []
    if hashtags:
        parts.append("Hashtags: " + " ".join(f"#{tag}" for tag in hashtags))
    if transcript:
        parts.append("Transcript: " + transcript)
    return "\n".join(parts).strip()[:6000]


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _tfidf_vectors(docs: list[str]) -> tuple[np.ndarray, str]:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )
    try:
        matrix = vectorizer.fit_transform(docs)
    except ValueError:
        # no usable tokens at all (e.g. emoji-only captions)
        return np.zeros((len(docs), 1), dtype=np.float32), "tfidf-empty"
    components = min(128, matrix.shape[0] - 1, matrix.shape[1] - 1)
    if components >= 2:
        matrix = TruncatedSVD(n_components=components, random_state=42).fit_transform(matrix)
    return normalize(matrix).astype(np.float32), "tfidf"


# --------------------------------------------------------------------------- #
# clustering
# --------------------------------------------------------------------------- #
def _merge_small_clusters(X: np.ndarray, labels: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 1:
        return labels
    labels = labels.copy()

    while True:
        counts = Counter(int(v) for v in labels)
        if len(counts) <= 1:
            break
        small = [cluster for cluster, size in counts.items() if size < min_size]
        if not small:
            break

        centroids: dict[int, np.ndarray] = {}
        for cluster in counts:
            centroid = X[labels == cluster].mean(axis=0)
            norm = float(np.linalg.norm(centroid))
            centroids[cluster] = centroid / norm if norm else centroid

        changed = False
        for cluster in small:
            if counts[cluster] >= min_size:
                continue
            others = [c for c in counts if c != cluster and counts[c] >= min_size]
            if not others:
                break
            for index in np.where(labels == cluster)[0]:
                similarities = [float(X[index] @ centroids[c]) for c in others]
                labels[index] = others[int(np.argmax(similarities))]
            changed = True
        if not changed:
            break

    unique = sorted({int(v) for v in labels})
    mapping = {old: new for new, old in enumerate(unique)}
    return np.array([mapping[int(v)] for v in labels], dtype=int)


def _choose_labels(
    X: np.ndarray,
    forced_k: int | None,
    max_clusters: int,
    min_cluster_size: int,
) -> tuple[np.ndarray, str]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = len(X)
    if n <= 1:
        return np.zeros(n, dtype=int), "single"

    if forced_k and forced_k > 0:
        k = max(1, min(forced_k, n))
        if k == 1:
            return np.zeros(n, dtype=int), "forced-k=1"
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        return _merge_small_clusters(X, labels, min_cluster_size), f"kmeans-k={k}"

    if n < 4:
        return np.zeros(n, dtype=int), "single"

    best_k = 1
    best_score = -1.0
    best_labels: np.ndarray | None = None
    upper = min(max_clusters, n // max(1, min_cluster_size))
    for k in range(2, upper + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        if len(set(int(v) for v in labels)) < 2:
            continue
        try:
            score = float(silhouette_score(X, labels, metric="cosine"))
        except Exception:  # noqa: BLE001
            continue
        if score > best_score + 1e-6:
            best_k, best_score, best_labels = k, score, labels

    if best_labels is None or best_score <= 0:
        return np.zeros(n, dtype=int), "single"

    merged = _merge_small_clusters(X, best_labels, min_cluster_size)
    return merged, f"kmeans-auto (k={best_k}, silhouette={best_score:.3f})"


def _cluster_confidence(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    confidence = np.zeros(len(X), dtype=np.float32)
    for cluster in sorted({int(v) for v in labels}):
        members = np.where(labels == cluster)[0]
        centroid = X[members].mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm:
            centroid = centroid / norm
        confidence[members] = X[members] @ centroid
    return confidence


# --------------------------------------------------------------------------- #
# naming
# --------------------------------------------------------------------------- #
def _heuristic_name(records: list[dict[str, Any]]) -> str:
    counter: Counter[str] = Counter()
    for record in records:
        for tag in record.get("hashtags") or []:
            if tag not in GENERIC_HASHTAGS:
                counter[tag] += 1
    words = [tag.replace("_", " ").title() for tag, _ in counter.most_common(2)]
    return " & ".join(words) if words else "Topic"


def _name_cluster(
    cfg,
    llm: LLM | None,
    records: list[dict[str, Any]],
    transcripts: list[str],
    member_positions: np.ndarray,
    record_positions: list[int],
    X: np.ndarray,
    labels: np.ndarray,
    cluster: int,
    sample_count: int,
    language: str,
) -> dict[str, Any]:
    """Name one cluster.

    ``member_positions`` index into the embedded/usable subset (rows of ``X``);
    ``record_positions[i]`` is the index in ``records`` for ``member_positions[i]``.
    """
    members = X[labels == cluster]
    centroid = members.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm:
        centroid = centroid / norm
    similarities = X[member_positions] @ centroid
    ordered = np.argsort(-similarities)[:sample_count]

    sample_lines: list[str] = []
    for local in ordered:
        record = records[record_positions[local]]
        caption = one_line(record.get("caption") or "", 260)
        transcript = one_line(transcripts[record_positions[local]], 260)
        line = f"- caption: {caption or '(none)'}"
        if transcript:
            line += f" | transcript: {transcript}"
        line += f" | author: @{record.get('author') or 'unknown'}"
        sample_lines.append(line)

    prompt = (
        f"These {len(member_positions)} saved videos were grouped because their captions/"
        "transcripts are about similar things.\n\n"
        + "\n".join(sample_lines)
        + "\n\nReturn ONE JSON object with this shape:\n"
        '{"name": "...", "description": "...", "tags": ["..."]}\n\n'
        "Rules:\n"
        '- "name": 2-4 words, Title Case, safe as a folder name, no emojis, no slashes, '
        f"max 40 characters, written in {language}.\n"
        '- "description": one sentence, max 140 characters.\n'
        '- "tags": 3 to 6 short lowercase keywords.\n'
        "Use only the samples above."
    )

    name = ""
    description = ""
    tags: list[str] = []
    if llm is not None and llm.available:
        answer = llm.chat_json(NAMING_SYSTEM, prompt, max_tokens=300)
        if answer:
            name = str(answer.get("name") or "").strip()
            description = str(answer.get("description") or "").strip()
            raw_tags = answer.get("tags") or []
            if isinstance(raw_tags, str):
                raw_tags = [part.strip() for part in raw_tags.split(",")]
            if isinstance(raw_tags, list):
                tags = [str(tag).strip().lower() for tag in raw_tags if str(tag).strip()][:6]

    if len(name) < 2:
        name = _heuristic_name([records[position] for position in record_positions])
    if not description:
        description = f"{len(member_positions)} saved videos about {name}."
    if not tags:
        tags = sorted(
            {
                tag
                for position in record_positions
                for tag in (records[position].get("hashtags") or [])
                if tag not in GENERIC_HASHTAGS
            }
        )[:6]

    return {
        "name": folder_name(name, fallback="Topic"),
        "description": one_line(description, 160),
        "tags": tags,
    }


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def run_classify(
    cfg,
    *,
    clusters: int | None = None,
    max_clusters: int | None = None,
    use_llm: bool = True,
) -> dict[str, Any]:
    banner("Classify videos into topics")

    records = load_records(cfg)
    if not records:
        warn("no posts found — run `python -m igvault collect` first")
        return {"topics": [], "items": {}}

    work_dir = cfg.path_for("paths.work")
    work_dir.mkdir(parents=True, exist_ok=True)

    transcripts = [_load_transcript(cfg, record) for record in records]
    docs = [_document(record, transcript) for record, transcript in zip(records, transcripts)]

    unsorted_name = folder_name(cfg.get("organize.unsorted_name", "Unsorted"), "Unsorted")
    language = cfg.get("classify.naming_language", "English")

    usable_positions = [index for index, doc in enumerate(docs) if doc.strip()]
    llm = LLM(cfg) if use_llm else None
    if use_llm and llm is not None and not llm.available:
        warn(f"LLM unavailable ({llm.explain_unavailable()}); using TF-IDF + hashtag names")

    items: dict[str, dict[str, Any]] = {
        record["shortcode"]: {
            "cluster": -1,
            "topic": unsorted_name,
            "confidence": 0.0,
            "tags": [],
            "reason": "no caption, hashtags or transcript",
        }
        for record in records
    }

    if use_llm and llm is not None and llm.available:
        result = _run_llm_classify(
            cfg, llm, records, transcripts, items,
            target=clusters if clusters and clusters > 0 else (
                cfg.get("classify.clusters") if isinstance(cfg.get("classify.clusters"), int) else None
            ),
            max_topics=max_clusters or int(cfg.get("classify.max_clusters", 12) or 12),
            language=language,
            unsorted_name=unsorted_name,
        )
        _print_summary(result)
        return result

    if not usable_positions:
        result = _write_result(cfg, records, items, [], "no-text", None)
        _print_summary(result)
        return result

    if len(usable_positions) == 1:
        labels = np.zeros(1, dtype=int)
        X = np.zeros((1, 2), dtype=np.float32)
        method = "single"
        confidence = np.ones(1, dtype=np.float32)
    else:
        X, vector_method = _embed(cfg, llm, [docs[position] for position in usable_positions])
        forced_k = clusters
        if forced_k is None:
            configured = cfg.get("classify.clusters", "auto")
            forced_k = configured if isinstance(configured, int) and configured > 0 else None
        labels, cluster_method = _choose_labels(
            X,
            forced_k=forced_k,
            max_clusters=max_clusters or int(cfg.get("classify.max_clusters", 12) or 12),
            min_cluster_size=int(cfg.get("classify.min_cluster_size", 3) or 1),
        )
        method = f"{vector_method}+{cluster_method}"
        confidence = _cluster_confidence(X, labels)

    sample_count = int(cfg.get("classify.naming_samples", 8) or 8)
    outlier_similarity = float(cfg.get("classify.outlier_similarity", 0.0) or 0.0)
    min_size = int(cfg.get("classify.min_cluster_size", 3) or 1)

    topics: list[dict[str, Any]] = []
    used_names: set[str] = set()
    cluster_count = len({int(v) for v in labels})

    for cluster in sorted({int(v) for v in labels}):
        positions = np.where(labels == cluster)[0]
        record_positions = [usable_positions[position] for position in positions]

        if cluster_count == 1:
            name = _heuristic_name([records[position] for position in record_positions])
            description = f"All {len(record_positions)} classified saved videos."
            tags = sorted(
                {
                    tag
                    for position in record_positions
                    for tag in (records[position].get("hashtags") or [])
                    if tag not in GENERIC_HASHTAGS
                }
            )[:6]
        else:
            naming = _name_cluster(
                cfg,
                llm if use_llm else None,
                records,
                transcripts,
                positions,
                record_positions,
                X,
                labels,
                cluster,
                sample_count,
                language,
            )
            name = naming["name"]
            description = naming["description"]
            tags = naming["tags"]

        base = name
        suffix = 2
        while name.lower() in used_names:
            name = f"{base} {suffix}"
            suffix += 1
        used_names.add(name.lower())

        topic = {
            "id": cluster,
            "name": name,
            "description": description,
            "tags": tags,
            "size": len(record_positions),
        }
        topics.append(topic)

        for position in positions:
            record = records[usable_positions[position]]
            sim = float(confidence[position])
            is_outlier = (
                outlier_similarity > 0
                and sim < outlier_similarity
                and len(record_positions) > min_size
            )
            items[record["shortcode"]] = {
                "cluster": cluster if not is_outlier else -1,
                "topic": name if not is_outlier else unsorted_name,
                "confidence": round(sim, 4),
                "tags": tags,
                "reason": "low similarity to topic centroid" if is_outlier else "",
            }

    if any(item["topic"] == unsorted_name for item in items.values()):
        topics.append(
            {
                "id": -1,
                "name": unsorted_name,
                "description": "Videos without enough text to classify confidently.",
                "tags": [],
                "size": sum(1 for item in items.values() if item["topic"] == unsorted_name),
            }
        )

    result = _write_result(cfg, records, items, topics, method, llm)
    _print_summary(result)
    return result


def _run_llm_classify(
    cfg,
    llm: LLM,
    records: list[dict[str, Any]],
    transcripts: list[str],
    items: dict[str, dict[str, Any]],
    *,
    target: int | None,
    max_topics: int,
    language: str,
    unsorted_name: str,
) -> dict[str, Any]:
    from .classify_llm import OTHER, assign_topics, build_taxonomy, describe_videos, refine_taxonomy

    descriptions = describe_videos(cfg, llm, records, transcripts)
    if not descriptions:
        warn("the LLM could not describe any video; nothing classified")
        return _write_result(cfg, records, items, [], "llm-failed", llm)

    import copy

    planner = copy.copy(llm)
    planner.chat_model = cfg.get("llm.taxonomy_model") or llm.chat_model
    topics = build_taxonomy(cfg, planner, descriptions, target=target, max_topics=max_topics, language=language)
    if not topics:
        warn("the LLM did not return a topic list; nothing classified")
        return _write_result(cfg, records, items, [], "llm-failed", llm)
    log(f"Topics: {', '.join(t['name'] for t in topics)}")

    log("Assigning videos to topics…")
    assignments = assign_topics(cfg, llm, topics, descriptions)

    # If many videos ended up in "Other", add folders for them and re-file those videos.
    leftovers = {sc: descriptions[sc] for sc, a in assignments.items() if a["topic"] == OTHER}
    if not target and not cfg.get("classify.topics") and len(leftovers) > max(5, 0.08 * len(assignments)):
        extra = refine_taxonomy(cfg, planner, topics, leftovers, max_topics=max_topics + 4, language=language)
        if extra:
            topics = topics + extra
            log(f"Added topics for leftovers: {', '.join(t['name'] for t in extra)}")
            assignments.update(assign_topics(cfg, llm, topics, leftovers))

    other_name = folder_name(OTHER, OTHER)
    names = {t["name"]: i for i, t in enumerate(topics)}
    for record in records:
        sc = record["shortcode"]
        desc = descriptions.get(sc)
        if not desc:
            continue  # stays Unsorted ("no caption, hashtags or transcript")
        item = items[sc]
        item["tags"] = desc["tags"]
        item["summary"] = desc["summary"]
        if desc.get("unclear"):
            item["reason"] = "not enough information to tell what the video is about"
            continue
        assigned = assignments.get(sc)
        if not assigned:
            item["reason"] = "the LLM did not assign a topic"
            continue
        topic = assigned["topic"]
        if topic == OTHER:
            item.update(cluster=len(topics), topic=other_name, confidence=assigned["confidence"],
                        reason="does not fit any main topic")
        else:
            item.update(cluster=names[topic], topic=topic, confidence=assigned["confidence"], reason="")

    counts = Counter(item["topic"] for item in items.values())
    out_topics: list[dict[str, Any]] = []
    for i, topic in enumerate(topics):
        if not counts.get(topic["name"]):
            continue
        members = [sc for sc, it in items.items() if it["topic"] == topic["name"]]
        tag_counts = Counter(tag for sc in members for tag in items[sc]["tags"])
        out_topics.append({
            "id": i,
            "name": topic["name"],
            "description": topic["description"],
            "tags": [tag for tag, _ in tag_counts.most_common(6)],
            "size": counts[topic["name"]],
        })
    if counts.get(other_name):
        out_topics.append({"id": len(topics), "name": other_name,
                           "description": "Videos that do not fit any of the main topics.",
                           "tags": [], "size": counts[other_name]})
    if counts.get(unsorted_name):
        out_topics.append({"id": -1, "name": unsorted_name,
                           "description": "Videos without enough information to classify.",
                           "tags": [], "size": counts[unsorted_name]})
    return _write_result(cfg, records, items, out_topics, "llm-describe+taxonomy+assign", llm)


def _embed(cfg, llm: LLM | None, docs: list[str]) -> tuple[np.ndarray, str]:
    vectors = llm.embed(docs) if llm is not None and llm.available else None
    if vectors:
        matrix = _normalize_rows(np.asarray(vectors, dtype=np.float32))
        return matrix, "embeddings"
    try:
        return _tfidf_vectors(docs)
    except ImportError:
        from .utils import die

        die("scikit-learn is required for the TF-IDF fallback. Run: pip install -r requirements.txt")


def _write_result(
    cfg,
    records: list[dict[str, Any]],
    items: dict[str, dict[str, Any]],
    topics: list[dict[str, Any]],
    method: str,
    llm: LLM | None,
) -> dict[str, Any]:
    result = {
        "generated_at": iso_now(),
        "method": method,
        "embedding_model": (
            cfg.get("llm.embedding_model") if llm is not None and llm.available else "tfidf"
        ),
        "chat_model": cfg.get("llm.chat_model") if llm is not None and llm.available else None,
        "topics": sorted(topics, key=lambda topic: (-topic["size"], topic["name"])),
        "items": items,
        "total_videos": len(records),
    }
    output = cfg.path_for("paths.work") / "classification.json"
    write_json(output, result)
    log(f"Wrote {output}")
    return result


def _print_summary(result: dict[str, Any]) -> None:
    topics = result.get("topics") or []
    total = result.get("total_videos") or 0
    log(f"Method: {result.get('method')}")
    log(f"{total} video(s) across {len([t for t in topics if t['id'] >= 0])} topic(s):")
    for topic in topics:
        log(f"  {topic['size']:4d}  {topic['name']}  —  {topic['description']}")

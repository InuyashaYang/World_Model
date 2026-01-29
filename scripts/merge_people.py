from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
SCORED_DIR = DATA_DIR / "scored"
PROFILES_DIR = DATA_DIR / "profiles"
OUT_DIR = DATA_DIR / "people"


def _load_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _name_from_filename(p: Path) -> str:
    # name.hash.score.json / name.hash.profile.json
    return p.name.split(".", 1)[0]


def _pick_latest_by_mtime(paths):
    return max(paths, key=lambda x: x.stat().st_mtime)


def _collect_latest(dir_path: Path, suffix: str) -> Dict[str, Path]:
    """
    以 name 为 key，选择同名多版本里 mtime 最新的一个。
    """
    groups: Dict[str, list[Path]] = {}
    for p in dir_path.glob(f"*.{suffix}.json"):
        name = _name_from_filename(p)
        groups.setdefault(name, []).append(p)
    return {name: _pick_latest_by_mtime(ps) for name, ps in groups.items()}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    score_map = _collect_latest(SCORED_DIR, "score")
    profile_map = _collect_latest(PROFILES_DIR, "profile")

    names = sorted(set(score_map.keys()) | set(profile_map.keys()))
    if not names:
        print("No inputs to merge.")
        return

    for name in names:
        score_p = score_map.get(name)
        prof_p = profile_map.get(name)

        score = _load_json(score_p) if score_p else None
        profile = _load_json(prof_p) if prof_p else None

        person: Dict[str, Any] = {
            "name": name,
            "score": score,  # 含大分/小分/完备度等
            "profile": profile,  # 标准化资料键值对
            "sources": {
                "score_file": str(score_p) if score_p else None,
                "profile_file": str(prof_p) if prof_p else None,
            },
        }

        out_path = OUT_DIR / f"{name}.person.json"
        out_path.write_text(
            json.dumps(person, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("written:", out_path.name)

    print("All done.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest: dict[str, object] = {"questions": {}, "common_core_sha256": None}
core_paths = [ROOT / f"问题{q}答案/q1_solver.py" for q in (1, 2, 3)]
core_hashes = [sha256(path) for path in core_paths]
if len(set(core_hashes)) != 1:
    raise RuntimeError(f"三问公共内核哈希不一致: {core_hashes}")
manifest["common_core_sha256"] = core_hashes[0]

for question in (1, 2, 3):
    source = ROOT / "outputs" / f"q{question}"
    target = ROOT / f"问题{question}答案"
    hashes: dict[str, str] = {}
    for path in sorted(source.iterdir()):
        if not path.is_file():
            continue
        destination = target / path.name
        shutil.copy2(path, destination)
        source_hash = sha256(path)
        target_hash = sha256(destination)
        if source_hash != target_hash:
            raise RuntimeError(f"同步后哈希不一致: {path.name}")
        hashes[path.name] = source_hash
    manifest["questions"][str(question)] = hashes

manifest_path = HERE / "正式交付文件清单.json"
manifest_path.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(manifest, ensure_ascii=False, indent=2))

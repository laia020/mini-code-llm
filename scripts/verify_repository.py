"""Check publishable files for large weights, caches and recognizable secrets."""

import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_results import verify_hashes


def verify():
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, capture_output=True, check=True,
    )
    paths = sorted(set(name.decode("utf-8") for name in result.stdout.split(b"\0") if name))
    forbidden_parts = {".venv", "venv", "__pycache__", ".cache", "cache", "checkpoints", "models", "gpu-backup"}
    forbidden_extensions = {".safetensors", ".bin", ".pt", ".pth", ".pem", ".key", ".zip", ".7z"}
    patterns = [
        re.compile(rb"hf_[A-Za-z0-9]{25,}"),
        re.compile(rb"gh[pousr]_[A-Za-z0-9]{25,}"),
        re.compile(rb"sk-(?:proj-)?[A-Za-z0-9_-]{30,}"),
        re.compile(rb"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----"),
    ]
    problems = []
    total = 0
    for name in paths:
        path = ROOT / name
        if not path.is_file():
            continue
        if forbidden_parts.intersection(path.relative_to(ROOT).parts) or path.suffix in forbidden_extensions or path.name.startswith((".env", "id_rsa", "id_ed25519")):
            problems.append(f"Arquivo não publicável: {name}")
        size = path.stat().st_size
        total += size
        if size > 10 * 1024 * 1024:
            problems.append(f"Arquivo acima de 10 MiB: {name}")
        # Report only filenames, never secret contents.
        if any(pattern.search(path.read_bytes()) for pattern in patterns):
            problems.append(f"Possível segredo: {name}")
    if problems:
        raise ValueError("\n".join(problems))
    return {"status": "PASS", "publishable_files": len(paths), "publishable_bytes": total, "immutable_artifacts": verify_hashes(), "secret_scan": "No matches in publishable files", "note": "Heuristic scan, plus path/size checks; ignored local caches are excluded."}


if __name__ == "__main__":
    try:
        print(json.dumps(verify(), indent=2))
    except (ValueError, AssertionError, subprocess.CalledProcessError) as error:
        print(f"REPOSITORY CHECK FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)

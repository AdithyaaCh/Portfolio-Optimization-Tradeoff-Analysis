from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def write_run_manifest(config: dict) -> Path:
    root = Path(config["_extension_dir"])
    config_path = Path(config["_config_path"])
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = "unavailable"
    packages = {}
    for name in ("numpy", "pandas", "scipy", "matplotlib", "seaborn", "PyYAML", "yfinance"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit_at_run": commit,
        "seed": config["project"]["seed"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "packages": packages,
    }
    output = root / "results" / "run_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output

#!/usr/bin/env python3
"""Keeps the job's data between GitHub runs and publishes the results.

  python ci/state.py restore   before the run: newest saved data from the release 'state'
                               (first run: the two price-history zips uploaded to the repository)
  python ci/state.py publish   after the run: results folder -> branch 'results' (replaced on every run)
  python ci/state.py save      after the run: data files -> new asset on the release 'state' (3 newest kept)

On GitHub it uses the 'gh' command and the automatic GITHUB_TOKEN (no secrets to set up).
For a local test pass --local-dir <folder>: saved data and the published branch then go to that folder instead.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RELEASE = "state"
KEEP_ASSETS = 3
KEEP_DAYS = 5
STATE_FILES = ["prices.pkl", "bench.pkl", "universe.csv", "store_meta.json", "corporate_actions.csv",
               "recommendations.db", "published.json"]
SEED_ZIPS = ["nifty500_ai_prices_part1.zip", "nifty500_ai_prices_part2.zip"]


def log(msg: str):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}Z  {msg}", flush=True)


def sh(cmd: list[str], check=True, capture=False, cwd=None) -> subprocess.CompletedProcess:
    shown = " ".join(c if "x-access-token" not in c else "<repo-url>" for c in cmd)
    log("$ " + shown)
    r = subprocess.run(cmd, cwd=cwd, text=True, capture_output=capture)
    if check and r.returncode != 0:
        err = (r.stderr or "").strip() if capture else ""
        raise RuntimeError(f"command failed ({r.returncode}): {shown} {err}")
    return r


def repo() -> str:
    r = os.environ.get("GITHUB_REPOSITORY")
    if not r:
        raise RuntimeError("GITHUB_REPOSITORY is not set (use --local-dir for a local test)")
    return r


def push_url() -> str:
    tok = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not tok:
        raise RuntimeError("GH_TOKEN is not set")
    return f"https://x-access-token:{tok}@github.com/{repo()}.git"


# ------------------------------------------------------------------------------------------------------------------
# saved data (release assets)
# ------------------------------------------------------------------------------------------------------------------
def list_assets(local: Path | None) -> list[str]:
    if local:
        d = local / "state"
        return sorted(p.name for p in d.glob("state_*.tar.gz")) if d.exists() else []
    r = sh(["gh", "release", "view", RELEASE, "--repo", repo(), "--json", "assets"], check=False, capture=True)
    if r.returncode != 0:
        return []                                  # release not created yet (first run)
    return sorted(a["name"] for a in json.loads(r.stdout).get("assets", [])
                  if a["name"].startswith("state_") and a["name"].endswith(".tar.gz"))


def download_asset(name: str, dest: Path, local: Path | None) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    if local:
        shutil.copy2(local / "state" / name, dest / name)
    else:
        sh(["gh", "release", "download", RELEASE, "--repo", repo(), "--pattern", name, "--dir", str(dest), "--clobber"])
    return dest / name


def seed_from_zips() -> bool:
    """First run: take prices_part1/2.pkl out of the two zips that were uploaded to the repository."""
    found = []
    for name in SEED_ZIPS:
        hits = [p for p in (ROOT / name, ROOT / "seed" / name) if p.exists()]
        if hits:
            found.append(hits[0])
    if len(found) < len(SEED_ZIPS):
        return False
    for z in found:
        with zipfile.ZipFile(z) as zf:
            for m in zf.namelist():
                base = m.rsplit("/", 1)[-1]
                if base.startswith("prices_part") and base.endswith(".pkl"):
                    with zf.open(m) as src, open(DATA / base, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    log(f"seed: {base} extracted from {z.name}")
    return any(DATA.glob("prices_part*.pkl"))


def restore_history_backup(local: Path | None):
    """If the saved data was lost, take the call history from the last published results (backup/)."""
    tmp = Path(tempfile.mkdtemp())
    try:
        if local:
            src = local / "results.git"
            if not src.exists():
                return
            url = str(src)
        else:
            url = push_url()
        r = sh(["git", "clone", "--quiet", "--depth", "1", "--branch", "results", url, str(tmp / "r")], check=False,
               capture=True)
        db = tmp / "r" / "backup" / "recommendations.db"
        if r.returncode == 0 and db.exists():
            shutil.copy2(db, DATA / "recommendations.db")
            log("call history restored from the results branch backup")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def cmd_restore(local: Path | None):
    DATA.mkdir(exist_ok=True)
    names = list_assets(local)
    if names:
        name = names[-1]
        tmp = Path(tempfile.mkdtemp())
        try:
            f = download_asset(name, tmp, local)
            with tarfile.open(f, "r:gz") as tf:
                members = [m for m in tf.getmembers() if m.isfile() and m.name in STATE_FILES]
                try:
                    tf.extractall(DATA, members=members, filter="data")
                except TypeError:                      # Python without extraction filters
                    tf.extractall(DATA, members=members)
            log(f"saved data restored from {name}: {', '.join(m.name for m in members)}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return
    log("no saved data yet (first run) - starting from the price-history zips in the repository")
    if not (DATA / "prices.pkl").exists() and not seed_from_zips():
        raise SystemExit("ERROR: price history not found. Upload nifty500_ai_prices_part1.zip and "
                         "nifty500_ai_prices_part2.zip to the repository (setup guide, Step 3) and run again.")
    if not (DATA / "recommendations.db").exists():
        restore_history_backup(local)


def cmd_save(local: Path | None):
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    name = f"state_{stamp}.tar.gz"
    tmp = Path(tempfile.mkdtemp())
    try:
        f = tmp / name
        with tarfile.open(f, "w:gz", compresslevel=6) as tf:
            for fn in STATE_FILES:
                p = DATA / fn
                if p.exists():
                    tf.add(p, arcname=fn)
        log(f"{name}: {f.stat().st_size / 1e6:.1f} MB")
        if local:
            d = local / "state"
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, d / name)
        else:
            if sh(["gh", "release", "view", RELEASE, "--repo", repo()], check=False, capture=True).returncode != 0:
                sh(["gh", "release", "create", RELEASE, "--repo", repo(), "--prerelease",
                    "--title", "Saved data of the daily job (automatic - please do not delete)",
                    "--notes", "Price history and call history used by the daily GitHub job. Updated automatically "
                               "after every run; the 3 newest copies are kept."])
            sh(["gh", "release", "upload", RELEASE, str(f), "--repo", repo(), "--clobber"])
        names = list_assets(local)
        for old in names[:-KEEP_ASSETS]:
            if local:
                (local / "state" / old).unlink(missing_ok=True)
            else:
                sh(["gh", "release", "delete-asset", RELEASE, old, "--repo", repo(), "--yes"], check=False)
            log(f"old copy removed: {old}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------------------------------------------------
# results branch
# ------------------------------------------------------------------------------------------------------------------
def cmd_publish(local: Path | None, results_dir: Path):
    status = json.load(open(results_dir / "status.json"))
    session = status["session_date"]
    if local:
        remote = local / "results.git"
        if not remote.exists():
            sh(["git", "init", "--quiet", "--bare", str(remote)])
        url = str(remote)
    else:
        url = push_url()
    tmp = Path(tempfile.mkdtemp())
    try:
        old, new = tmp / "old", tmp / "new"
        r = sh(["git", "clone", "--quiet", "--depth", "1", "--branch", "results", url, str(old)], check=False,
               capture=True)
        shutil.copytree(results_dir, new)
        days = {p.name for p in (new / "d").iterdir() if p.is_dir()}
        if r.returncode == 0 and (old / "d").exists():                # keep a few earlier days (CDN cache safety)
            for p in sorted((old / "d").iterdir(), reverse=True):
                if p.is_dir() and p.name not in days and len(days) < KEEP_DAYS:
                    shutil.copytree(p, new / "d" / p.name)
                    days.add(p.name)
        git = ["git", "-C", str(new)]
        sh(git + ["init", "--quiet"])
        sh(git + ["checkout", "--quiet", "-b", "results"])
        sh(git + ["add", "-A"])
        sh(git + ["-c", "user.name=github-actions[bot]",
                  "-c", "user.email=41898282+github-actions[bot]@users.noreply.github.com",
                  "-c", "commit.gpgsign=false",
                  "commit", "--quiet", "-m", f"AI results for session {session}"])
        sh(git + ["-c", "push.negotiate=false", "push", "--quiet", "--force", url, "results:results"])
        log(f"published session {session} on branch 'results' (days kept: {', '.join(sorted(days))})")
        json.dump({"session_date": session, "published_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                  open(DATA / "published.json", "w"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["restore", "save", "publish"])
    ap.add_argument("--results-dir", default=str(ROOT / "build" / "results"))
    ap.add_argument("--local-dir", help="(testing) keep saved data and the results branch in this folder")
    a = ap.parse_args()
    local = Path(a.local_dir).resolve() if a.local_dir else None
    if a.action == "restore":
        cmd_restore(local)
    elif a.action == "save":
        cmd_save(local)
    else:
        cmd_publish(local, Path(a.results_dir))


if __name__ == "__main__":
    main()

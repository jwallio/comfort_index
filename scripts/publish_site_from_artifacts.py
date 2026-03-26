from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path


def github_request(url: str, token: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "comfortwx-site-publisher",
        },
    )
    with urllib.request.urlopen(request) as response:
        return response.read()


def latest_artifact_id(repository: str, token: str, artifact_name: str) -> int | None:
    payload = github_request(
        f"https://api.github.com/repos/{repository}/actions/artifacts?per_page=100",
        token,
    )
    data = json.loads(payload.decode("utf-8"))
    artifacts = [
        artifact
        for artifact in data.get("artifacts", [])
        if artifact.get("name") == artifact_name and artifact.get("expired") is False
    ]
    if not artifacts:
        return None
    artifacts.sort(key=lambda artifact: artifact.get("created_at", ""), reverse=True)
    return artifacts[0]["id"]


def download_and_extract_artifact(
    repository: str,
    token: str,
    artifact_id: int,
    target_dir: Path,
) -> None:
    archive_bytes = github_request(
        f"https://api.github.com/repos/{repository}/actions/artifacts/{artifact_id}/zip",
        token,
    )
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as handle:
        handle.write(archive_bytes)
        archive_path = Path(handle.name)
    try:
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(target_dir)
    finally:
        archive_path.unlink(missing_ok=True)


def copy_tree_if_exists(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    destination.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        target = destination / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)
    return True


def write_fallback_site(site_dir: Path, message: str) -> None:
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "index.html").write_text(
        (
            "<html><head><meta charset='utf-8'><title>Comfort Index Archive</title></head>"
            "<body><h1>Comfort Index Archive</h1>"
            f"<p>{message}</p></body></html>"
        ),
        encoding="utf-8",
    )


def main() -> None:
    repository = os.environ["GITHUB_REPOSITORY"]
    token = os.environ["GH_TOKEN"]
    archive_artifact_name = os.environ.get("ARCHIVE_ARTIFACT_NAME", "comfortwx-archive")
    verification_artifact_name = os.environ.get(
        "VERIFICATION_ARTIFACT_NAME",
        "comfortwx-verification-benchmark",
    )

    site_dir = Path("site")
    archive_dir = Path("archive_artifact")
    verification_dir = Path("verification_artifact")

    shutil.rmtree(site_dir, ignore_errors=True)
    shutil.rmtree(archive_dir, ignore_errors=True)
    shutil.rmtree(verification_dir, ignore_errors=True)

    archive_id = latest_artifact_id(repository, token, archive_artifact_name)
    if archive_id is None:
        write_fallback_site(site_dir, "No non-expired archive artifact is currently available.")
    else:
        download_and_extract_artifact(repository, token, archive_id, archive_dir)
        copied = (
            copy_tree_if_exists(archive_dir / "output" / "archive", site_dir)
            or copy_tree_if_exists(archive_dir / "archive", site_dir)
        )
        if not copied:
            write_fallback_site(site_dir, "The latest archive artifact did not contain an archive directory.")

    verification_id = latest_artifact_id(repository, token, verification_artifact_name)
    if verification_id is not None:
        download_and_extract_artifact(repository, token, verification_id, verification_dir)
        verification_site = site_dir / "verification"
        copied = (
            copy_tree_if_exists(
                verification_dir / "output" / "verification_site" / "latest",
                verification_site,
            )
            or copy_tree_if_exists(
                verification_dir / "verification_site" / "latest",
                verification_site,
            )
        )
        if not copied:
            verification_site.mkdir(parents=True, exist_ok=True)

    (site_dir / ".nojekyll").touch()


if __name__ == "__main__":
    main()

"""Build a distributable source ZIP without environments, tokens, models or data."""
from pathlib import Path
import argparse
import zipfile

ROOT = Path(__file__).resolve().parent.parent
DIRECTORIES = {"jk_step", "jk_engine", "frontend", "assets", "scripts", "tests", "configs", "docs"}
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", "node_modules", "tensorboard", ".cache", ".git"}
SUFFIXES = {".py", ".js", ".css", ".html", ".json", ".toml", ".txt", ".md", ".bat", ".ps1", ".sh", ".png", ".jpg", ".jpeg", ".svg", ".ico", ".woff", ".woff2", ".ttf", ".license"}


def release_files():
    for source in sorted(ROOT.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(ROOT)
        if any(part in EXCLUDE_PARTS for part in relative.parts):
            continue
        if len(relative.parts) == 1:
            if source.name in {"LICENSE", ".gitignore", ".python-version"} or source.suffix in SUFFIXES:
                if source.name not in {"requirements.local.txt"}:
                    yield source, relative
        elif relative.parts[0] in DIRECTORIES and source.suffix in SUFFIXES:
            yield source, relative


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dist/JK-Step_MR-FlowDPO-0.1.0.zip")
    args = parser.parse_args()
    output = Path(args.output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    files = list(release_files())
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source, relative in files:
            archive.write(source, str(Path("JK-Step_MR-FlowDPO") / relative))
    print(f"{output}: {len(files)} source files; no environment, checkpoint, dataset or session included.")


if __name__ == "__main__":
    main()

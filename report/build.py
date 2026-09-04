"""Build and validate the bilingual LaTeX portfolio reports."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent
BUILD_DIR = REPORT_DIR / "build"
PDF_DIR = REPORT_DIR / "pdf"
OUTPUT_NAMES = {
    "fr": "leaf-classification-portfolio-fr.pdf",
    "en": "leaf-classification-portfolio-en.pdf",
}
KNOWN_LATEXMK = (
    Path.home() / "AppData/Local/Programs/MiKTeX/miktex/bin/x64/latexmk.exe",
    Path("C:/Program Files/MiKTeX/miktex/bin/x64/latexmk.exe"),
)
KNOWN_PDFLATEX = tuple(path.with_name("pdflatex.exe") for path in KNOWN_LATEXMK)
FATAL_LOG_PATTERNS = {
    "undefined references": r"LaTeX Warning:.*(?:Reference|Citation).*undefined",
    "rerun warning": r"Rerun to get (?:cross-references|outlines) right",
    "missing file": r"(?:LaTeX Error: File .* not found|Package .* Error: File .* not found)",
    "overfull box": r"Overfull \\hbox|Overfull \\vbox",
}


def resolve_latexmk() -> Path:
    configured = os.environ.get("LATEXMK")
    candidates = [Path(configured).expanduser()] if configured else []
    discovered = shutil.which("latexmk")
    if discovered:
        candidates.append(Path(discovered))
    candidates.extend(KNOWN_LATEXMK)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "latexmk was not found. Set LATEXMK, add it to PATH, or install MiKTeX "
        "in one of the supported locations."
    )


def resolve_pdflatex() -> Path:
    configured = os.environ.get("PDFLATEX")
    candidates = [Path(configured).expanduser()] if configured else []
    discovered = shutil.which("pdflatex")
    if discovered:
        candidates.append(Path(discovered))
    candidates.extend(KNOWN_PDFLATEX)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "PDFLaTeX fallback was not found. Set PDFLATEX, add it to PATH, or install MiKTeX."
    )


def selected_languages(value: str) -> tuple[str, ...]:
    return ("fr", "en") if value == "all" else (value,)


def clean(languages: tuple[str, ...]) -> None:
    for language in languages:
        shutil.rmtree(BUILD_DIR / language, ignore_errors=True)
        (PDF_DIR / OUTPUT_NAMES[language]).unlink(missing_ok=True)


def check_sources(language: str) -> None:
    section_dir = REPORT_DIR / language / "sections"
    section_paths = sorted(section_dir.glob("[0-9][0-9]-*.tex"))
    if len(section_paths) != 7:
        raise RuntimeError(
            f"{language}: expected 7 section files, found {len(section_paths)}"
        )
    content = "\n".join(path.read_text(encoding="utf-8") for path in section_paths)
    figure_paths = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", content)
    if len(figure_paths) != 36 or len(set(figure_paths)) != 36:
        raise RuntimeError(
            f"{language}: expected 36 unique referenced figures, found "
            f"{len(figure_paths)} references and {len(set(figure_paths))} unique paths"
        )
    missing = [
        path
        for path in figure_paths
        if not (REPORT_DIR / language / path).resolve().is_file()
    ]
    if missing:
        raise RuntimeError(f"{language}: missing figures: {', '.join(missing)}")


def validate_log(language: str) -> None:
    log_path = BUILD_DIR / language / "main.log"
    log = log_path.read_text(encoding="utf-8", errors="replace")
    problems = [
        label
        for label, pattern in FATAL_LOG_PATTERNS.items()
        if re.search(pattern, log)
    ]
    if problems:
        raise RuntimeError(
            f"{language}: LaTeX log validation failed: {', '.join(problems)}"
        )


def run_latex(language: str, latexmk: Path, output_dir: Path) -> None:
    language_dir = REPORT_DIR / language
    if os.name != "nt" or shutil.which("perl"):
        command = [
            str(latexmk),
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            f"-outdir={output_dir}",
            "main.tex",
        ]
        subprocess.run(command, cwd=language_dir, check=True)
        return

    # MiKTeX's latexmk wrapper requires Perl, which is not part of a default
    # Windows install. Three PDFLaTeX passes provide the same stable TOC and
    # hyperlinks without weakening CI, where latexmk remains the primary path.
    pdflatex = resolve_pdflatex()
    command = [
        str(pdflatex),
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        f"-output-directory={output_dir}",
        "main.tex",
    ]
    for _ in range(3):
        subprocess.run(command, cwd=language_dir, check=True)


def build(language: str, latexmk: Path, strict: bool) -> None:
    check_sources(language)
    output_dir = BUILD_DIR / language
    output_dir.mkdir(parents=True, exist_ok=True)
    run_latex(language, latexmk, output_dir)
    produced_pdf = output_dir / "main.pdf"
    if not produced_pdf.is_file():
        raise RuntimeError(
            f"{language}: LaTeX build completed without producing {produced_pdf}"
        )
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(produced_pdf, PDF_DIR / OUTPUT_NAMES[language])
    if strict:
        validate_log(language)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--language",
        choices=("fr", "en", "all"),
        default="all",
        help="language edition to process (default: all)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail on missing figures, undefined references, rerun warnings, or overfull boxes",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="remove selected build products and exit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    languages = selected_languages(args.language)
    if args.clean:
        clean(languages)
        return 0
    try:
        latexmk = resolve_latexmk()
        for language in languages:
            build(language, latexmk, args.check)
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

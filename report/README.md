# Bilingual portfolio report

This directory contains revised French and English portfolio editions of the
Leaf Classification report. Both editions preserve the scientific protocol,
results, and conclusions of the April 2026 archived submission. The portfolio
revision covers language, layout, and translation.

## Build

Requirements:

- Python 3.10 or newer
- `latexmk` with a working PDFLaTeX installation (MiKTeX or TeX Live)
- Perl when using `latexmk` on Windows; without Perl, the script automatically
  performs three PDFLaTeX passes

Build both editions:

```console
python report/build.py --language all
```

Regenerate the localized figure set from the checked dataset, canonical report
values, and French diagnostic assets before rebuilding:

```console
python report/generate_localized_figures.py
```

Run strict source and LaTeX-log validation:

```console
python report/build.py --language all --check
```

Remove intermediates and generated PDFs:

```console
python report/build.py --language all --clean
```

Set `LATEXMK` to the executable path when it is neither on `PATH` nor in a
standard MiKTeX location.

Final PDFs are written to:

- `report/pdf/leaf-classification-portfolio-fr.pdf`
- `report/pdf/leaf-classification-portfolio-en.pdf`

The 36 figures referenced by each edition live in the corresponding
`report/figures/fr` or `report/figures/en` directory.

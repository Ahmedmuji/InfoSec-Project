# How to compile this paper on Overleaf

## 1. Upload

1. Go to https://www.overleaf.com and create a **new blank project**.
2. Click **Upload Project** (top-left, "New Project" → "Upload Project")
   and drag-and-drop a ZIP of this entire `paper/` folder, **or** upload
   the two items individually:
   - `privfed_tcn_paper.tex`
   - the `figures/` folder (all 6 PNGs)

## 2. Set the main document and compiler

- In Overleaf's **Menu** (top-left), set:
  - **Main document**: `privfed_tcn_paper.tex`
  - **Compiler**: `pdfLaTeX`
  - **TeX Live version**: 2023 or newer (default is fine)

## 3. Compile

Click **Recompile**. The first build is slow because IEEEtran +
TikZ + pgfplots are loaded. Subsequent builds take ~5 seconds.

## 4. Folder layout expected by the .tex file

```
paper/
├── privfed_tcn_paper.tex
└── figures/
    ├── fig1_privacy_accuracy_tradeoff.png
    ├── fig2_convergence_comparison.png
    ├── fig3_communication_cost.png
    ├── fig4_per_class_f1.png
    ├── fig6_confusion_matrix.png
    └── fig7_headline_metrics.png
```

## 5. Optional: replace PNGs with PDF figures

If you want vector-quality figures in the published PDF, drop the `.pdf`
versions of each figure (already produced in
`privfed_tcn/results/figures/`) into the same `figures/` folder and
change the `\includegraphics{...png}` filenames to `.pdf`. PDF figures
are sharper at any zoom level for IEEE submission.

## 6. Common Overleaf issues

- **"File not found" for figures**: make sure the `figures/` folder is
  inside the same project root as the `.tex` file, not nested deeper.
- **TikZ shape errors**: the `\usetikzlibrary{...}` block at the top of
  the file is required; do not delete it.
- **Bibliography**: bibliography entries are inline in `thebibliography`
  so no `.bib` file is needed.

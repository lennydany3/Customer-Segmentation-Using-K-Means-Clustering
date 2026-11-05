# Project report and conference paper

- `CustomerSegmentML_Project_Report.docx` / `.pdf`: 39-page course report (21CSC305P Machine Learning).
- `CustomerSegmentML_IEEE_Conference_Paper.docx` / `.pdf`: 5-page IEEE two-column conference paper.

Both describe the current project: a Tamil Nadu gift shop (Key Chains, Bouquet, Miniature Toys) whose customers are clustered with K-Means on Age and Repeat_Purchases, evaluated on the three files in `dataset/` (70, 85 and 100 customers).

All numbers come from running `backend/model.py` on those files. Accuracy is measured against the hidden groups that `dataset/generate_datasets.py` used to create the customers (they are not stored in the CSV files).

Reproduce the experiments and charts (needs scikit-learn, pandas, scipy, matplotlib), from the repository root:

```bash
python report/scripts/experiments.py . report/scripts/results.json
```

```bash
python report/scripts/figures.py
```

`figures.py` reads `results.json` from its own folder and writes the charts to `report/scripts/figs/`.

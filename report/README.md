# Project report and conference paper

- `CustomerSegmentML_Project_Report.docx` / `.pdf`: 40-page course report (21CSC305P Machine Learning).
- `CustomerSegmentML_IEEE_Conference_Paper.docx` / `.pdf`: 4-page IEEE two-column conference paper.

All numbers come from running `backend/model.py` on the Kaggle Mall Customers dataset (200 rows).

Reproduce the experiments and charts (needs scikit-learn, pandas, matplotlib):

```bash
python scripts/exp.py path/to/Mall_Customers.csv scripts/results.json
```

`scripts/figs.py` expects `results.json` next to it and the CSV at `../data/mall.csv` relative to the script folder.

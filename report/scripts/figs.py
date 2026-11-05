import json, sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

W = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(W, "figs"); os.makedirs(OUT, exist_ok=True)
R = json.load(open(os.path.join(W, "results.json")))
df = pd.read_csv(os.path.join(W, "..", "data", "mall.csv"))
inc = df["Annual Income (k$)"].values.astype(float); sp = df["Spending Score (1-100)"].values.astype(float)
age = df["Age"].values.astype(float)
X = np.c_[inc, sp]; Xs = StandardScaler().fit_transform(X)
PRODUCT = "CustomerSegment ML"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3, "savefig.dpi": 200})
C5 = ["#2F6FDE", "#16A34A", "#9333EA", "#EA580C", "#0891B2"]
OURS = "#1F5FD6"; LIGHT = "#A9C4F5"; GREY = "#C9CED6"; BAD = "#E8A1A1"

def save(name):
    plt.tight_layout(); plt.savefig(os.path.join(OUT, name), bbox_inches="tight", facecolor="white"); plt.close()

# 1 raw scatter
plt.figure(figsize=(7, 4.4))
plt.scatter(inc, sp, s=28, c="#5B7DB8", alpha=0.8, edgecolor="white", lw=0.5)
plt.xlabel("Annual income (k$)"); plt.ylabel("Spending score (1 to 100)")
save("f_raw_scatter.png")

# 2 histograms
fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
for a, v, t, c in zip(ax, [age, inc, sp], ["Age (years)", "Annual income (k$)", "Spending score"], ["#7C9CE0", "#7CC79A", "#E0A97C"]):
    a.hist(v, bins=18, color=c, edgecolor="white"); a.set_xlabel(t); a.set_ylabel("Customers")
    a.axvline(v.mean(), color="#333", ls="--", lw=1); a.text(v.mean(), a.get_ylim()[1] * 0.92, f" mean {v.mean():.1f}", fontsize=9)
save("f_hist.png")

# 3 correlation heatmap
cm = np.corrcoef(np.c_[age, inc, sp].T); lab = ["Age", "Income", "Spending"]
plt.figure(figsize=(4.6, 3.8)); im = plt.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
plt.xticks(range(3), lab); plt.yticks(range(3), lab); plt.grid(False)
for i in range(3):
    for j in range(3): plt.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=11)
plt.colorbar(im, fraction=0.046); save("f_corr.png")

# 4 scaling before/after
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
ax[0].scatter(inc, sp, s=14, c="#5B7DB8"); ax[0].set_title("Before scaling (original units)", fontsize=11)
ax[0].set_xlabel("Income (k$)"); ax[0].set_ylabel("Spending score")
ax[1].scatter(Xs[:, 0], Xs[:, 1], s=14, c="#16A34A"); ax[1].set_title("After StandardScaler (z-scores)", fontsize=11)
ax[1].set_xlabel("Income (z)"); ax[1].set_ylabel("Spending (z)"); ax[1].axhline(0, c="#888", lw=0.8); ax[1].axvline(0, c="#888", lw=0.8)
save("f_scaling.png")

ks = R["ks"]
# 5 elbow
plt.figure(figsize=(7, 4))
plt.plot(ks, R["wcss"], "-o", c=OURS, lw=2.2)
for k, w in zip(ks, R["wcss"]): plt.annotate(f"{w:.1f}", (k, w), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
plt.scatter([5], [R["wcss"][3]], s=220, facecolors="none", edgecolors="#DC2626", lw=2)
plt.annotate("elbow at K = 5", (5, R["wcss"][3]), xytext=(5.6, 150), arrowprops=dict(arrowstyle="->", color="#DC2626"), color="#DC2626")
plt.xlabel("Number of clusters K"); plt.ylabel("WCSS (inertia, scaled space)"); save("f_elbow.png")

# 6 silhouette vs K
plt.figure(figsize=(7, 3.8))
cols = [OURS if k == 5 else LIGHT for k in ks]
b = plt.bar(ks, R["sil"], color=cols)
for k, s in zip(ks, R["sil"]): plt.text(k, s + 0.01, f"{s:.3f}", ha="center", fontsize=9)
plt.ylim(0, 0.65); plt.xlabel("Number of clusters K"); plt.ylabel("Mean silhouette score"); save("f_sil_k.png")

# 7 DB and CH vs K
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(ks, R["db"], "-o", c="#EA580C"); ax[0].set_xlabel("K"); ax[0].set_ylabel("Davies-Bouldin (lower is better)")
ax[0].scatter([5], [R["db"][3]], s=160, facecolors="none", edgecolors=OURS, lw=2)
ax[1].plot(ks, R["ch"], "-o", c="#16A34A"); ax[1].set_xlabel("K"); ax[1].set_ylabel("Calinski-Harabasz (higher is better)")
save("f_db_ch.png")

# 8 final clusters
pts = R["points"]; lab5 = np.array([p["cluster"] for p in pts])
names = {c["cluster_id"]: c["label"].split(" (")[0] for c in R["clusters"]}
plt.figure(figsize=(7.6, 5))
for c in range(5):
    m = lab5 == c
    plt.scatter(inc[m], sp[m], s=30, c=C5[c], label=f"C{c}: {names[c]} ({m.sum()})", edgecolor="white", lw=0.5)
for c in R["clusters"]:
    plt.scatter(*c["centroid"], marker="D", s=140, c="#111", edgecolor="white", lw=1.5, zorder=5)
plt.xlabel("Annual income (k$)"); plt.ylabel("Spending score (1 to 100)")
plt.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
save("f_clusters.png")

# 9 silhouette plot per sample
km = KMeans(5, random_state=42, n_init=10).fit(Xs); sv = silhouette_samples(Xs, km.labels_)
plt.figure(figsize=(7, 4.6)); y = 10
for c in range(5):
    v = np.sort(sv[km.labels_ == c]); plt.fill_betweenx(np.arange(y, y + len(v)), 0, v, color=C5[c], alpha=0.85)
    plt.text(-0.06, y + len(v) / 2, f"C{c}", fontsize=9); y += len(v) + 8
plt.axvline(sv.mean(), c="#DC2626", ls="--"); plt.text(sv.mean() + 0.01, 5, f"mean = {sv.mean():.3f}", color="#DC2626")
plt.yticks([]); plt.xlabel("Silhouette coefficient s(i)"); plt.grid(axis="y", visible=False); save("f_sil_samples.png")

# 10 profiles
cl = R["clusters"]; idx = np.arange(5); w = 0.27
plt.figure(figsize=(8.5, 4))
plt.bar(idx - w, [c["avg_income"] for c in cl], w, label="Avg income (k$)", color="#2F6FDE")
plt.bar(idx, [c["avg_spending"] for c in cl], w, label="Avg spending score", color="#16A34A")
plt.bar(idx + w, [c["avg_age"] for c in cl], w, label="Avg age (years)", color="#EA580C")
plt.xticks(idx, [f"C{c['cluster_id']}\n{names[c['cluster_id']]}" for c in cl], fontsize=8.5)
plt.legend(fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False); save("f_profiles.png")

# 11 sizes
plt.figure(figsize=(6, 4.2))
plt.pie([c["customer_count"] for c in cl], labels=[f"C{c['cluster_id']} {names[c['cluster_id']]}" for c in cl], colors=C5,
        autopct="%1.1f%%", startangle=90, wedgeprops=dict(edgecolor="white", width=0.45), textprops=dict(fontsize=9))
plt.axis("equal"); save("f_sizes.png")

# 12 model comparison (silhouette)
rows = R["rows"]
nice = {
    "KMeans proposed (scaled, k-means++, n_init=10)": f"{PRODUCT} (K-Means, proposed)",
    "Agglomerative (Ward)": "Agglomerative (Ward)", "Gaussian Mixture": "Gaussian Mixture Model",
    "BIRCH": "BIRCH", "MiniBatch KMeans": "Mini-Batch K-Means",
    "KMeans random init, single run (avg 20 seeds)": "K-Means random init, 1 run",
    "Agglomerative (average)": "Agglomerative (average)", "Agglomerative (complete)": "Agglomerative (complete)",
    "Agglomerative (single)": "Agglomerative (single)", "DBSCAN (eps=0.5)": "DBSCAN (eps = 0.5)",
    "Rule-based income tiers": "Rule-based income tiers", "Income-only KMeans": "Income-only K-Means",
    "Random assignment (avg 20)": "Random assignment",
}
rs = sorted(rows, key=lambda r: r["sil"])
plt.figure(figsize=(8.6, 5.6))
labels = [nice[r["model"]] for r in rs]; vals = [r["sil"] for r in rs]
colors = [OURS if "proposed" in l else (BAD if v < 0.3 else GREY) for l, v in zip(labels, vals)]
bars = plt.barh(labels, vals, color=colors)
for bb, v, l in zip(bars, vals, labels):
    plt.text(max(v, 0) + 0.008, bb.get_y() + bb.get_height() / 2, f"{v:.3f}", va="center", fontsize=9,
             fontweight="bold" if "proposed" in l else "normal", color=OURS if "proposed" in l else "#333")
for t in plt.gca().get_yticklabels():
    if "proposed" in t.get_text(): t.set_fontweight("bold"); t.set_color(OURS)
plt.axvline(0, c="#555", lw=0.8); plt.xlim(-0.1, 0.66); plt.xlabel("Mean silhouette score (higher is better), K = 5")
plt.grid(axis="y", visible=False); save("f_cmp_sil.png")

# 13 CH comparison
rs2 = sorted(rows, key=lambda r: r["ch"])
plt.figure(figsize=(8.6, 5.6))
labels = [nice[r["model"]] for r in rs2]; vals = [r["ch"] for r in rs2]
colors = [OURS if "proposed" in l else (BAD if v < 100 else GREY) for l, v in zip(labels, vals)]
bars = plt.barh(labels, vals, color=colors)
for bb, v, l in zip(bars, vals, labels):
    plt.text(v + 3, bb.get_y() + bb.get_height() / 2, f"{v:.1f}", va="center", fontsize=9,
             fontweight="bold" if "proposed" in l else "normal", color=OURS if "proposed" in l else "#333")
for t in plt.gca().get_yticklabels():
    if "proposed" in t.get_text(): t.set_fontweight("bold"); t.set_color(OURS)
plt.xlim(0, 290); plt.xlabel("Calinski-Harabasz index (higher is better), K = 5"); plt.grid(axis="y", visible=False)
save("f_cmp_ch.png")

# 14 stability
s_pp, s_rand = [], []
for seed in range(30):
    from sklearn.metrics import silhouette_score
    s_pp.append(silhouette_score(Xs, KMeans(5, random_state=seed, n_init=10).fit_predict(Xs)))
    s_rand.append(silhouette_score(Xs, KMeans(5, init="random", n_init=1, random_state=seed).fit_predict(Xs)))
plt.figure(figsize=(7.2, 3.8))
bp = plt.boxplot([s_rand, s_pp], tick_labels=["K-Means random init,\nsingle run", f"{PRODUCT}\n(k-means++, n_init = 10)"],
                 patch_artist=True, widths=0.5)
for patch, c in zip(bp["boxes"], [GREY, OURS]): patch.set_facecolor(c)
for i, s in enumerate([s_rand, s_pp]): plt.scatter(np.full(len(s), i + 1) + np.random.default_rng(1).normal(0, 0.04, len(s)), s, s=10, c="#222", zorder=3)
plt.ylabel("Silhouette over 30 random seeds"); save("f_stability.png")
json.dump(dict(pp_mean=float(np.mean(s_pp)), pp_std=float(np.std(s_pp)), r_mean=float(np.mean(s_rand)), r_std=float(np.std(s_rand)), r_min=float(np.min(s_rand))), open(os.path.join(W, "stab.json"), "w"))

# 15 convergence (manual Lloyd's with k-means++ seed 42 centers)
rng = np.random.default_rng(7)
init = Xs[rng.choice(len(Xs), 5, replace=False)]
C = init.copy(); hist = []
for it in range(12):
    d = ((Xs[:, None, :] - C[None]) ** 2).sum(-1); l = d.argmin(1); hist.append(d.min(1).sum())
    C = np.array([Xs[l == j].mean(0) if (l == j).any() else C[j] for j in range(5)])
plt.figure(figsize=(7, 3.6)); plt.plot(range(1, 13), hist, "-o", c=OURS)
plt.xlabel("Lloyd iteration"); plt.ylabel("WCSS"); save("f_converge.png")

# 16 decision regions
gx, gy = np.meshgrid(np.linspace(10, 142, 300), np.linspace(0, 102, 300))
sc = StandardScaler().fit(X); g = sc.transform(np.c_[gx.ravel(), gy.ravel()])
from matplotlib.colors import ListedColormap
zz = km.predict(g).reshape(gx.shape)
# map km labels to app cluster ids via centroids
cent_app = np.array([c["centroid"] for c in cl]); cent_km = sc.inverse_transform(km.cluster_centers_)
mp = {i: int(np.argmin(((cent_app - cent_km[i]) ** 2).sum(1))) for i in range(5)}
zz = np.vectorize(mp.get)(zz)
plt.figure(figsize=(7.4, 4.8)); plt.contourf(gx, gy, zz, levels=np.arange(-0.5, 5.5), cmap=ListedColormap(C5), alpha=0.18)
for c in range(5):
    m = lab5 == c; plt.scatter(inc[m], sp[m], s=18, c=C5[c])
for q, t in [((95, 85), "A"), ((90, 15), "B"), ((25, 80), "C"), ((25, 20), "D"), ((55, 50), "E")]:
    plt.scatter(*q, marker="*", s=300, c="#E11D48", edgecolor="white", zorder=6); plt.text(q[0] + 2, q[1] + 2, t, fontsize=11, fontweight="bold")
plt.xlabel("Annual income (k$)"); plt.ylabel("Spending score"); plt.grid(False); save("f_regions.png")

# ---------- diagrams ----------
def box(ax, x, y, w, h, text, fc, fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="#333", lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold", color="#1f2937")
def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#222"))
PAL = ["#BFD7FF", "#DCD3FF", "#FFF3A8", "#FFC9D6", "#C9F2D2", "#FFD8A8", "#BDEFF5"]

def fig_ax(w, h, xl, yl):
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(0, xl); ax.set_ylim(0, yl); ax.axis("off"); return ax

# architecture
ax = fig_ax(11, 5.2, 11, 5.2)
box(ax, 0.2, 3.7, 1.9, 0.9, "Browser\n(user)", PAL[0])
box(ax, 2.6, 3.7, 2.4, 0.9, "Next.js 16 Dashboard\nReact + Recharts", PAL[1])
box(ax, 5.6, 3.7, 2.2, 0.9, "Typed API client\n(lib/api.ts)", PAL[2])
box(ax, 8.4, 3.7, 2.4, 0.9, "FastAPI service\n(main.py)", PAL[3])
box(ax, 8.4, 2.1, 2.4, 0.9, "Pydantic schemas\n(schemas.py)", PAL[6])
box(ax, 5.4, 2.1, 2.6, 0.9, "CustomerSegmentation\nModel (model.py)", PAL[4])
box(ax, 2.6, 2.1, 2.4, 0.9, "StandardScaler +\nK-Means (sklearn)", PAL[5])
box(ax, 0.2, 2.1, 1.9, 0.9, "Persona +\nstrategy rules", PAL[0])
box(ax, 0.6, 0.4, 9.8, 0.8, "In-memory state: uploaded DataFrame, fitted scaler, fitted K-Means, cluster metadata", "#E5E7EB", 10)
arrow(ax, 2.1, 4.15, 2.6, 4.15); arrow(ax, 5.0, 4.15, 5.6, 4.15); arrow(ax, 7.8, 4.15, 8.4, 4.15)
arrow(ax, 9.6, 3.7, 9.6, 3.0); arrow(ax, 8.4, 2.55, 8.0, 2.55); arrow(ax, 5.4, 2.55, 5.0, 2.55); arrow(ax, 2.6, 2.55, 2.1, 2.55)
arrow(ax, 6.7, 2.1, 6.7, 1.2)
ax.text(8.1, 3.45, "HTTP/JSON", fontsize=8.5, ha="center")
save("d_arch.png")

# flowchart
ax = fig_ax(11, 3.6, 11, 3.6)
top = ["Upload\nCSV", "Detect &\nvalidate columns", "Drop invalid\nrows", "StandardScaler\n(z-score)", "Elbow sweep\nK = 2 to 8", "Pick K by\nmax silhouette"]
bot = ["Fit K-Means\n(k-means++, n=10)", "Inverse-scale\ncentroids", "Label personas\n& strategies", "Render charts\n& tables", "Predict new\ncustomer", "Download\nsegmented CSV"]
for i, t in enumerate(top): box(ax, 0.15 + i * 1.8, 2.4, 1.5, 0.95, t, PAL[i % 7], 9)
for i, t in enumerate(bot): box(ax, 0.15 + i * 1.8, 0.4, 1.5, 0.95, t, PAL[(i + 3) % 7], 9)
for i in range(5): arrow(ax, 1.65 + i * 1.8, 2.87, 1.95 + i * 1.8, 2.87); arrow(ax, 1.65 + i * 1.8, 0.87, 1.95 + i * 1.8, 0.87)
ax.annotate("", (0.9, 1.35), (9.9, 2.4), arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#222", connectionstyle="angle,angleA=-90,angleB=180,rad=0"))
save("d_flow.png")

# k-means loop
ax = fig_ax(10, 3.4, 10, 3.4)
box(ax, 0.2, 1.2, 1.9, 1.0, "Scaled data\nX (n x 2)", PAL[0])
box(ax, 2.6, 1.2, 1.9, 1.0, "k-means++\nseeding", PAL[1])
box(ax, 5.0, 2.1, 2.0, 0.95, "Assign to nearest\ncentroid, Eq. (4)", PAL[2], 9)
box(ax, 5.0, 0.3, 2.0, 0.95, "Recompute means,\nEq. (5)", PAL[3], 9)
box(ax, 7.7, 1.2, 2.1, 1.0, "Converged?\nbest of 10 runs", PAL[4], 9)
arrow(ax, 2.1, 1.7, 2.6, 1.7); arrow(ax, 4.5, 1.9, 5.0, 2.5); arrow(ax, 5.6, 2.1, 5.6, 1.25); arrow(ax, 6.4, 1.25, 6.4, 2.1)
arrow(ax, 7.0, 0.8, 7.7, 1.5); ax.text(6.5, 1.6, "repeat until stable", fontsize=9, style="italic")
save("d_kmeans.png")

# K-sweep / model selection funnel
ax = fig_ax(10, 2.6, 10, 2.6)
items = ["K = 2 ... 8\ncandidates", "WCSS\n(elbow)", "Silhouette", "Davies-Bouldin\n& Calinski-H.", "Selected\nK = 5"]
for i, t in enumerate(items): box(ax, 0.2 + i * 1.98, 0.8, 1.6, 1.0, t, PAL[i], 9.5)
for i in range(4): arrow(ax, 1.8 + i * 1.98, 1.3, 2.18 + i * 1.98, 1.3)
save("d_select.png")

# API sequence
ax = fig_ax(10, 4.6, 10, 4.6)
for x, t, c in [(1.2, "Dashboard", PAL[1]), (5.0, "FastAPI", PAL[3]), (8.6, "Model", PAL[4])]:
    box(ax, x - 0.8, 3.8, 1.6, 0.6, t, c, 10); ax.plot([x, x], [0.2, 3.8], ls="--", c="#888", lw=1)
msgs = [(1.2, 5.0, 3.4, "POST /api/upload (text/csv)"), (5.0, 8.6, 3.1, "load_dataset()"), (1.2, 5.0, 2.6, "GET /api/elbow"),
        (5.0, 8.6, 2.3, "compute_elbow() -> recommended K"), (1.2, 5.0, 1.8, "POST /api/cluster {k}"), (5.0, 8.6, 1.5, "train(k)"),
        (1.2, 5.0, 1.0, "POST /api/predict {income, score}"), (5.0, 8.6, 0.7, "predict() -> segment + strategy")]
for x1, x2, y, t in msgs:
    arrow(ax, x1, y, x2, y); ax.text((x1 + x2) / 2, y + 0.07, t, ha="center", fontsize=8.5)
save("d_seq.png")

# appendix: test/phases strip
ax = fig_ax(10, 1.6, 10, 1.6)
for i, t in enumerate(["health", "upload", "dataset", "elbow", "cluster", "predict"]): box(ax, 0.2 + i * 1.65, 0.35, 1.3, 0.8, t, PAL[i], 10)
for i in range(5): arrow(ax, 1.5 + i * 1.65, 0.75, 1.85 + i * 1.65, 0.75)
save("d_tests.png")
print("done", sorted(os.listdir(OUT)))

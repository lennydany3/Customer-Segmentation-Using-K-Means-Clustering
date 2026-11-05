import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

W = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(W, "..", ".."))
OUT = os.path.join(W, "figs"); os.makedirs(OUT, exist_ok=True)
R = json.load(open(os.path.join(W, "results.json")))
F100 = R["files"]["tn_gift_shop_100.csv"]
df = pd.read_csv(os.path.join(REPO, "dataset", "tn_gift_shop_100.csv"))
age = df.Age.values.astype(float); rep = df.Repeat_Purchases.values.astype(float)
X = np.c_[age, rep]; sc = StandardScaler().fit(X); Xs = sc.transform(X)
PRODUCT = "CustomerSegment ML"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "savefig.dpi": 200})
C5 = ["#2F6FDE", "#16A34A", "#9333EA", "#EA580C", "#0891B2"]
OURS = "#1F5FD6"; LIGHT = "#A9C4F5"; GREY = "#C9CED6"; BAD = "#E8A1A1"
PCOL = {"Key Chains": "#2F6FDE", "Bouquet": "#E11D48", "Miniature Toys": "#F59E0B"}


def save(name):
    plt.tight_layout(); plt.savefig(os.path.join(OUT, name), bbox_inches="tight", facecolor="white"); plt.close()


cl = F100["clusters"]; names = {c["cluster_id"]: c["label"] for c in cl}
lab5 = np.array([p["cluster"] for p in F100["points"]])

# raw scatter coloured by product
plt.figure(figsize=(7, 4.4))
for p, c in PCOL.items():
    m = df.Product_Type == p; plt.scatter(age[m], rep[m], s=34, c=c, label=p, edgecolor="white", lw=0.5, alpha=0.85)
plt.xlabel("Age (years)"); plt.ylabel("Repeat purchases"); plt.legend(frameon=False, fontsize=9); save("f_raw.png")

# distributions
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
ax[0].hist(age, bins=16, color="#7C9CE0", edgecolor="white"); ax[0].set_xlabel("Age (years)"); ax[0].set_ylabel("Customers")
ax[1].hist(rep, bins=range(0, 21), color="#7CC79A", edgecolor="white"); ax[1].set_xlabel("Repeat purchases")
pc = df.Product_Type.value_counts(); ax[2].bar(pc.index, pc.values, color=[PCOL[p] for p in pc.index]); ax[2].set_ylabel("Customers")
ax[2].tick_params(axis="x", labelsize=8.5); ax[2].grid(axis="x", visible=False)
for i, v in enumerate(pc.values): ax[2].text(i, v + 0.8, str(v), ha="center", fontsize=9)
save("f_hist.png")

# districts
dc = df.District.value_counts()
plt.figure(figsize=(8.5, 3.6)); plt.bar(dc.index, dc.values, color="#8DA9E8"); plt.xticks(rotation=60, ha="right", fontsize=8.5)
plt.ylabel("Customers"); plt.grid(axis="x", visible=False); save("f_districts.png")

# scaling
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
ax[0].scatter(age, rep, s=14, c="#5B7DB8"); ax[0].set_title("Before scaling (original units)", fontsize=11); ax[0].set_xlabel("Age (years)"); ax[0].set_ylabel("Repeat purchases")
ax[1].scatter(Xs[:, 0], Xs[:, 1], s=14, c="#16A34A"); ax[1].set_title("After StandardScaler (z-scores)", fontsize=11)
ax[1].set_xlabel("Age (z)"); ax[1].set_ylabel("Repeat purchases (z)"); ax[1].axhline(0, c="#888", lw=.8); ax[1].axvline(0, c="#888", lw=.8)
save("f_scaling.png")

ks = F100["ks"]
plt.figure(figsize=(7, 4)); plt.plot(ks, F100["wcss"], "-o", c=OURS, lw=2.2)
for kk, w in zip(ks, F100["wcss"]): plt.annotate(f"{w:.1f}", (kk, w), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
i5 = ks.index(5); plt.scatter([5], [F100["wcss"][i5]], s=220, facecolors="none", edgecolors="#DC2626", lw=2)
plt.annotate("elbow at K = 5", (5, F100["wcss"][i5]), xytext=(5.6, max(F100["wcss"]) * 0.55), arrowprops=dict(arrowstyle="->", color="#DC2626"), color="#DC2626")
plt.xlabel("Number of clusters K"); plt.ylabel("WCSS (inertia, scaled space)"); save("f_elbow.png")

plt.figure(figsize=(7, 3.8)); plt.bar(ks, F100["sil_k"], color=[OURS if kk == 5 else LIGHT for kk in ks])
for kk, s in zip(ks, F100["sil_k"]): plt.text(kk, s + 0.01, f"{s:.3f}", ha="center", fontsize=9)
plt.ylim(0, 0.8); plt.xlabel("Number of clusters K"); plt.ylabel("Mean silhouette score"); save("f_sil_k.png")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(ks, F100["db_k"], "-o", c="#EA580C"); ax[0].set_xlabel("K"); ax[0].set_ylabel("Davies-Bouldin (lower is better)")
ax[1].plot(ks, F100["ch_k"], "-o", c="#16A34A"); ax[1].set_xlabel("K"); ax[1].set_ylabel("Calinski-Harabasz (higher is better)")
save("f_db_ch.png")

# accuracy across K for all three files
plt.figure(figsize=(7, 3.8))
for (f, v), c in zip(R["files"].items(), ["#16A34A", "#9333EA", OURS]):
    plt.plot(v["ks"], [a * 100 for a in v["acc_k"]], "-o", c=c, label=f"{f.replace('.csv', '')} (true K = {v['true_k']})")
    j = v["ks"].index(v["rec_k"]); plt.scatter([v["rec_k"]], [v["acc_k"][j] * 100], s=160, facecolors="none", edgecolors=c, lw=2)
plt.xlabel("Number of clusters K"); plt.ylabel("Accuracy vs hidden groups (%)"); plt.ylim(30, 105); plt.legend(fontsize=8.5, frameon=False); save("f_acc_k.png")

# final clusters
plt.figure(figsize=(7.6, 5))
for c in range(5):
    m = lab5 == c; plt.scatter(age[m], rep[m], s=30, c=C5[c], label=f"C{c}: {names[c]} ({m.sum()})", edgecolor="white", lw=0.5)
for c in cl: plt.scatter(*c["centroid"], marker="D", s=140, c="#111", edgecolor="white", lw=1.5, zorder=5)
plt.xlabel("Age (years)"); plt.ylabel("Repeat purchases")
plt.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False); save("f_clusters.png")

# silhouette samples
km = KMeans(5, random_state=42, n_init=10).fit(Xs); sv = silhouette_samples(Xs, km.labels_)
plt.figure(figsize=(7, 4.6)); y0 = 4
for c in range(5):
    v = np.sort(sv[km.labels_ == c]); plt.fill_betweenx(np.arange(y0, y0 + len(v)), 0, v, color=C5[c], alpha=0.85)
    plt.text(-0.07, y0 + len(v) / 2, f"C{c}", fontsize=9); y0 += len(v) + 4
plt.axvline(sv.mean(), c="#DC2626", ls="--"); plt.text(sv.mean() + 0.01, 2, f"mean = {sv.mean():.3f}", color="#DC2626")
plt.yticks([]); plt.xlabel("Silhouette coefficient s(i)"); plt.grid(axis="y", visible=False); save("f_sil_samples.png")

# profiles
idx = np.arange(5); w = 0.38
fig, ax1 = plt.subplots(figsize=(8.5, 4))
ax1.bar(idx - w / 2, [c["avg_age"] for c in cl], w, color="#2F6FDE", label="Avg age (years)"); ax1.set_ylabel("Average age")
ax2 = ax1.twinx(); ax2.bar(idx + w / 2, [c["avg_repeats"] for c in cl], w, color="#16A34A", label="Avg repeat purchases"); ax2.set_ylabel("Average repeat purchases")
ax2.grid(False); ax1.set_xticks(idx); ax1.set_xticklabels([f"C{c['cluster_id']}\n{c['label']}" for c in cl], fontsize=8)
fig.legend(loc="upper center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.04)); save("f_profiles.png")

# product mix per cluster
mix = pd.crosstab(lab5, df.Product_Type, normalize="index") * 100
plt.figure(figsize=(8.5, 4)); bottom = np.zeros(5)
for p in ["Key Chains", "Miniature Toys", "Bouquet"]:
    v = mix[p].reindex(range(5)).fillna(0).values; plt.bar(idx, v, bottom=bottom, color=PCOL[p], label=p)
    for i in range(5):
        if v[i] > 8: plt.text(i, bottom[i] + v[i] / 2, f"{v[i]:.0f}%", ha="center", va="center", fontsize=8.5, color="white")
    bottom += v
plt.xticks(idx, [f"C{c['cluster_id']}\n{c['label']}" for c in cl], fontsize=8); plt.ylabel("Share of cluster (%)")
plt.legend(ncol=3, frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 1.12)); plt.grid(axis="x", visible=False); save("f_mix.png")

plt.figure(figsize=(6, 4.2))
plt.pie([c["customer_count"] for c in cl], labels=[f"C{c['cluster_id']} {c['label']}" for c in cl], colors=C5, autopct="%1.0f%%",
        startangle=90, wedgeprops=dict(edgecolor="white", width=0.45), textprops=dict(fontsize=9)); plt.axis("equal"); save("f_sizes.png")

# convergence
rng = np.random.default_rng(3); C = Xs[rng.choice(len(Xs), 5, replace=False)].copy(); hist = []
for it in range(10):
    d = ((Xs[:, None, :] - C[None]) ** 2).sum(-1); l = d.argmin(1); hist.append(d.min(1).sum())
    C = np.array([Xs[l == j].mean(0) if (l == j).any() else C[j] for j in range(5)])
plt.figure(figsize=(7, 3.6)); plt.plot(range(1, 11), hist, "-o", c=OURS); plt.xlabel("Lloyd iteration"); plt.ylabel("WCSS"); save("f_converge.png")

# decision regions
gx, gy = np.meshgrid(np.linspace(14, 66, 300), np.linspace(-0.5, 21, 300)); g = sc.transform(np.c_[gx.ravel(), gy.ravel()])
cent_app = np.array([c["centroid"] for c in cl]); cent_km = sc.inverse_transform(km.cluster_centers_)
mp = {i: int(np.argmin(((cent_app - cent_km[i]) ** 2).sum(1))) for i in range(5)}
zz = np.vectorize(mp.get)(km.predict(g).reshape(gx.shape))
plt.figure(figsize=(7.4, 4.8)); plt.contourf(gx, gy, zz, levels=np.arange(-0.5, 5.5), cmap=ListedColormap(C5), alpha=0.18)
for c in range(5):
    m = lab5 == c; plt.scatter(age[m], rep[m], s=18, c=C5[c])
for q, t in [((20, 16), "A"), ((24, 3), "B"), ((37, 9), "C"), ((57, 2), "D"), ((55, 16), "E")]:
    plt.scatter(*q, marker="*", s=300, c="#E11D48", edgecolor="white", zorder=6); plt.text(q[0] + 0.8, q[1] + 0.6, t, fontsize=11, fontweight="bold")
plt.xlabel("Age (years)"); plt.ylabel("Repeat purchases"); plt.grid(False); save("f_regions.png")

# comparisons
NICE = {"proposed": f"{PRODUCT} (K-Means, proposed)", "gmm": "Gaussian Mixture Model", "ward": "Agglomerative (Ward)",
        "complete": "Agglomerative (complete)", "average": "Agglomerative (average)", "random1": "K-Means random init, 1 run",
        "birch": "BIRCH", "dbscan": "DBSCAN (eps = 0.3)", "single": "Agglomerative (single)", "age_only": "Age-only K-Means",
        "repeats_only": "Repeats-only K-Means", "age_bands": "Rule-based age bands", "product_rule": "Group by product type",
        "random": "Random assignment"}
rows = [r for r in R["rows"] if r["model"] in NICE]


def cmp_chart(key, fname, xlabel, fmt, bad, xlim, scale=1.0):
    rs = sorted(rows, key=lambda r: r[key])
    plt.figure(figsize=(8.6, 5.8)); labels = [NICE[r["model"]] for r in rs]; vals = [r[key] * scale for r in rs]
    colors = [OURS if "proposed" in l else (BAD if v < bad else GREY) for l, v in zip(labels, vals)]
    bars = plt.barh(labels, vals, color=colors)
    for bb, v, l in zip(bars, vals, labels):
        plt.text(max(v, 0) + (xlim[1] - xlim[0]) * 0.012, bb.get_y() + bb.get_height() / 2, fmt.format(v), va="center", fontsize=9,
                 fontweight="bold" if "proposed" in l else "normal", color=OURS if "proposed" in l else "#333")
    for t in plt.gca().get_yticklabels():
        if "proposed" in t.get_text(): t.set_fontweight("bold"); t.set_color(OURS)
    plt.axvline(0, c="#555", lw=0.8); plt.xlim(*xlim); plt.xlabel(xlabel); plt.grid(axis="y", visible=False); save(fname)


cmp_chart("sil", "f_cmp_sil.png", "Mean silhouette score (higher is better), K = 5", "{:.4f}", 0.3, (-0.15, 0.8))
cmp_chart("acc", "f_cmp_acc.png", "Accuracy against the hidden customer groups (%), K = 5", "{:.0f}%", 70, (0, 112), 100)
cmp_chart("ch", "f_cmp_ch.png", "Calinski-Harabasz index (higher is better), K = 5", "{:.1f}", 100, (0, 380))

st = R["stab"]
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.8))
for a, key, lab in [(ax[0], ("r", "pp"), "Silhouette"), (ax[1], ("acc_r", "acc_pp"), "Accuracy")]:
    data = [st[key[0]], st[key[1]]] if lab == "Silhouette" else [[v * 100 for v in st[key[0]]], [v * 100 for v in st[key[1]]]]
    bp = a.boxplot(data, tick_labels=["K-Means random\ninit, 1 run", f"{PRODUCT}\n(k-means++, 10 runs)"], patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], [GREY, OURS]): patch.set_facecolor(c)
    for i, s in enumerate(data): a.scatter(np.full(len(s), i + 1) + np.random.default_rng(1).normal(0, 0.04, len(s)), s, s=10, c="#222", zorder=3)
    a.set_ylabel(lab + (" (%)" if lab == "Accuracy" else "") + " over 30 seeds"); a.tick_params(axis="x", labelsize=8.5)
save("f_stability.png")


# ---------- diagrams
def box(ax, x, y, w, h, text, fc, fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="#333", lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold", color="#1f2937")
def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#222"))
PAL = ["#BFD7FF", "#DCD3FF", "#FFF3A8", "#FFC9D6", "#C9F2D2", "#FFD8A8", "#BDEFF5"]
def fig_ax(w, h):
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off"); return ax

ax = fig_ax(11, 5.2)
box(ax, 0.2, 3.7, 1.9, 0.9, "Browser\n(shop owner)", PAL[0]); box(ax, 2.6, 3.7, 2.4, 0.9, "Next.js 16 Dashboard\nReact + Recharts", PAL[1])
box(ax, 5.6, 3.7, 2.2, 0.9, "Typed API client\n(lib/api.ts)", PAL[2]); box(ax, 8.4, 3.7, 2.4, 0.9, "FastAPI service\n(main.py)", PAL[3])
box(ax, 8.4, 2.1, 2.4, 0.9, "Pydantic schemas\n(schemas.py)", PAL[6]); box(ax, 5.4, 2.1, 2.6, 0.9, "CustomerSegmentation\nModel (model.py)", PAL[4])
box(ax, 2.6, 2.1, 2.4, 0.9, "StandardScaler +\nK-Means (sklearn)", PAL[5]); box(ax, 0.2, 2.1, 1.9, 0.9, "Persona, product\n& district summary", PAL[0], 9)
box(ax, 0.6, 0.4, 9.8, 0.8, "In-memory state: uploaded customers, fitted scaler, fitted K-Means, cluster metadata", "#E5E7EB", 10)
for a in [(2.1, 4.15, 2.6, 4.15), (5.0, 4.15, 5.6, 4.15), (7.8, 4.15, 8.4, 4.15), (9.6, 3.7, 9.6, 3.0), (8.4, 2.55, 8.0, 2.55), (5.4, 2.55, 5.0, 2.55), (2.6, 2.55, 2.1, 2.55), (6.7, 2.1, 6.7, 1.2)]: arrow(ax, *a)
ax.text(8.1, 3.45, "HTTP/JSON", fontsize=8.5, ha="center"); save("d_arch.png")

ax = fig_ax(11, 3.6)
top = ["Upload\nCSV", "Detect &\nvalidate columns", "Drop invalid\nrows", "StandardScaler\n(age, repeats)", "Elbow sweep\nK = 2 to 8", "Pick K by\nmax silhouette"]
bot = ["Fit K-Means\n(k-means++, n=10)", "Inverse-scale\ncentroids", "Personas, top\nproduct & district", "Render charts\n& tables", "Predict new\ncustomer", "Download\nsegmented CSV"]
for i, t in enumerate(top): box(ax, 0.15 + i * 1.8, 2.4, 1.5, 0.95, t, PAL[i % 7], 9)
for i, t in enumerate(bot): box(ax, 0.15 + i * 1.8, 0.4, 1.5, 0.95, t, PAL[(i + 3) % 7], 9)
for i in range(5): arrow(ax, 1.65 + i * 1.8, 2.87, 1.95 + i * 1.8, 2.87); arrow(ax, 1.65 + i * 1.8, 0.87, 1.95 + i * 1.8, 0.87)
ax.annotate("", (0.9, 1.35), (9.9, 2.4), arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#222", connectionstyle="angle,angleA=-90,angleB=180,rad=0"))
save("d_flow.png")

ax = fig_ax(10, 3.4)
box(ax, 0.2, 1.2, 1.9, 1.0, "Scaled data\n[age, repeats]", PAL[0]); box(ax, 2.6, 1.2, 1.9, 1.0, "k-means++\nseeding", PAL[1])
box(ax, 5.0, 2.1, 2.0, 0.95, "Assign to nearest\ncentroid, Eq. (4)", PAL[2], 9); box(ax, 5.0, 0.3, 2.0, 0.95, "Recompute means,\nEq. (5)", PAL[3], 9)
box(ax, 7.7, 1.2, 2.1, 1.0, "Converged?\nbest of 10 runs", PAL[4], 9)
arrow(ax, 2.1, 1.7, 2.6, 1.7); arrow(ax, 4.5, 1.9, 5.0, 2.5); arrow(ax, 5.6, 2.1, 5.6, 1.25); arrow(ax, 6.4, 1.25, 6.4, 2.1); arrow(ax, 7.0, 0.8, 7.7, 1.5)
ax.text(6.5, 1.6, "repeat until stable", fontsize=9, style="italic"); save("d_kmeans.png")

ax = fig_ax(10, 2.6)
for i, t in enumerate(["K = 2 ... 8\ncandidates", "WCSS\n(elbow)", "Silhouette", "Davies-Bouldin\n& Calinski-H.", "Selected\nK = 5"]): box(ax, 0.2 + i * 1.98, 0.8, 1.6, 1.0, t, PAL[i], 9.5)
for i in range(4): arrow(ax, 1.8 + i * 1.98, 1.3, 2.18 + i * 1.98, 1.3)
save("d_select.png")

ax = fig_ax(10, 4.6)
for x, t, c in [(1.2, "Dashboard", PAL[1]), (5.0, "FastAPI", PAL[3]), (8.6, "Model", PAL[4])]:
    box(ax, x - 0.8, 3.8, 1.6, 0.6, t, c, 10); ax.plot([x, x], [0.2, 3.8], ls="--", c="#888", lw=1)
for x1, x2, yy, t in [(1.2, 5.0, 3.4, "POST /api/upload (text/csv)"), (5.0, 8.6, 3.1, "load_dataset()"), (1.2, 5.0, 2.6, "GET /api/elbow"),
                      (5.0, 8.6, 2.3, "compute_elbow() -> recommended K"), (1.2, 5.0, 1.8, "POST /api/cluster {k}"), (5.0, 8.6, 1.5, "train(k)"),
                      (1.2, 5.0, 1.0, "POST /api/predict {age, repeat_purchases}"), (5.0, 8.6, 0.7, "predict() -> segment + strategy")]:
    arrow(ax, x1, yy, x2, yy); ax.text((x1 + x2) / 2, yy + 0.07, t, ha="center", fontsize=8.5)
save("d_seq.png")
print("done", len(os.listdir(OUT)))

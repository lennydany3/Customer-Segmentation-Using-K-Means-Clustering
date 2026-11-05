"""Runs the project's own model and the comparison methods on the gift-shop datasets."""
import json, os, sys, time
REPO = sys.argv[1]
sys.path.insert(0, os.path.join(REPO, "backend")); sys.path.insert(0, os.path.join(REPO, "dataset"))
import numpy as np, pandas as pd
from model import CustomerSegmentationModel, _cluster_accuracy
import generate_datasets as G
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score

out = {"files": {}}
for file, (seed, sizes) in G.DATASETS.items():
    df = pd.read_csv(os.path.join(REPO, "dataset", file))
    truth_df = G.make(seed, sizes)
    assert truth_df[["Name", "Age", "Repeat_Purchases"]].equals(df[["Name", "Age", "Repeat_Purchases"]])
    y = pd.factorize(truth_df["True_Segment"])[0]
    m = CustomerSegmentationModel(); summ = m.load_dataset(df)
    ks, wcss, rk = m.compute_elbow()
    t = time.time(); res = m.train(rk); dt = time.time() - t
    X = m.get_features(); Xs = StandardScaler().fit_transform(X)
    sweep = {"sil_k": [], "db_k": [], "ch_k": [], "acc_k": []}
    for k in ks:
        l = KMeans(k, random_state=42, n_init=10).fit_predict(Xs)
        sweep["sil_k"].append(round(silhouette_score(Xs, l), 4)); sweep["db_k"].append(round(davies_bouldin_score(Xs, l), 4))
        sweep["ch_k"].append(round(calinski_harabasz_score(Xs, l), 2)); sweep["acc_k"].append(round(_cluster_accuracy(y, l), 4))
    pred = m.df["Cluster"].values
    out["files"][file] = dict(n=len(df), true_k=int(len(set(y))), rec_k=rk, ks=ks, wcss=wcss, **sweep,
                              sil=res["silhouette_score"], acc=round(_cluster_accuracy(y, pred), 4),
                              ari=round(adjusted_rand_score(y, pred), 4), iters=int(m.model.n_iter_), train_ms=round(dt * 1000, 1),
                              clusters=res["clusters"], points=res["points"], products=df.Product_Type.value_counts().to_dict(),
                              gender=df.Gender.value_counts().to_dict(), districts=int(df.District.nunique()),
                              age=[float(df.Age.min()), float(df.Age.max()), round(float(df.Age.mean()), 1), round(float(df.Age.std()), 1)],
                              rep=[float(df.Repeat_Purchases.min()), float(df.Repeat_Purchases.max()), round(float(df.Repeat_Purchases.mean()), 1), round(float(df.Repeat_Purchases.std()), 1)],
                              corr=round(float(np.corrcoef(df.Age, df.Repeat_Purchases)[0, 1]), 3), true_labels=[int(v) for v in y],
                              true_names=list(pd.factorize(truth_df["True_Segment"])[1]))
    if file == "tn_gift_shop_100.csv":
        main = dict(df=df, X=X, Xs=Xs, y=y, m=m, k=rk)

df, X, Xs, y, m, k = (main[x] for x in ["df", "X", "Xs", "y", "m", "k"])
preds = {}
for q in [(20, 16), (24, 3), (37, 9), (57, 2), (55, 16)]:
    p = m.predict(*q); preds[str(q)] = p["label"]
out["predictions"] = preds


def ev(name, lab):
    lab = np.asarray(lab); n = len(set(lab))
    if n < 2:
        return dict(model=name, k=n, sil=None, db=None, ch=None, acc=round(_cluster_accuracy(y, lab), 3), ari=0.0)
    return dict(model=name, k=n, sil=round(silhouette_score(Xs, lab), 4), db=round(davies_bouldin_score(Xs, lab), 3),
                ch=round(calinski_harabasz_score(Xs, lab), 1), acc=round(_cluster_accuracy(y, lab), 3), ari=round(adjusted_rand_score(y, lab), 3))


def avg(name, labs):
    rs = [ev(name, l) for l in labs]
    return {kk: (round(float(np.mean([r[kk] for r in rs])), 3) if kk not in ("model", "k") else rs[0][kk]) for kk in rs[0]}


rng = np.random.default_rng(42)
prod_codes = pd.factorize(df.Product_Type)[0]
rows = [
    ev("proposed", m.df["Cluster"].values),
    ev("gmm", GaussianMixture(k, random_state=42).fit_predict(Xs)),
    ev("ward", AgglomerativeClustering(k, linkage="ward").fit_predict(Xs)),
    ev("complete", AgglomerativeClustering(k, linkage="complete").fit_predict(Xs)),
    ev("average", AgglomerativeClustering(k, linkage="average").fit_predict(Xs)),
    ev("minibatch", MiniBatchKMeans(k, random_state=0, n_init=1, batch_size=10).fit_predict(Xs)),
    avg("random1", [KMeans(k, init="random", n_init=1, random_state=s).fit_predict(Xs) for s in range(20)]),
    ev("birch", Birch(n_clusters=k, threshold=0.3).fit_predict(Xs)),
    ev("single", AgglomerativeClustering(k, linkage="single").fit_predict(Xs)),
    ev("dbscan", DBSCAN(eps=0.3, min_samples=5).fit_predict(Xs)),
    ev("unscaled", KMeans(k, random_state=42, n_init=10).fit_predict(X * np.array([1.0, 1.0]))),
    ev("age_only", KMeans(k, random_state=42, n_init=10).fit_predict(Xs[:, [0]])),
    ev("repeats_only", KMeans(k, random_state=42, n_init=10).fit_predict(Xs[:, [1]])),
    ev("age_bands", pd.cut(df.Age, [0, 30, 45, 200], labels=False).values),
    ev("product_rule", prod_codes),
    avg("random", [rng.integers(0, k, len(Xs)) for _ in range(20)]),
]
out["rows"] = rows
# stability
s_pp, s_r, a_pp, a_r = [], [], [], []
for s in range(30):
    l1 = KMeans(k, random_state=s, n_init=10).fit_predict(Xs); l2 = KMeans(k, init="random", n_init=1, random_state=s).fit_predict(Xs)
    s_pp.append(silhouette_score(Xs, l1)); s_r.append(silhouette_score(Xs, l2)); a_pp.append(_cluster_accuracy(y, l1)); a_r.append(_cluster_accuracy(y, l2))
labs = [KMeans(k, random_state=s, n_init=10).fit_predict(Xs) for s in range(10)]
labr = [KMeans(k, init="random", random_state=s, n_init=1).fit_predict(Xs) for s in range(10)]
out["stab"] = dict(pp=s_pp, r=s_r, acc_pp=a_pp, acc_r=a_r,
                   ari_pp=float(np.mean([adjusted_rand_score(labs[0], l) for l in labs[1:]])),
                   ari_r=float(np.mean([adjusted_rand_score(labr[0], l) for l in labr[1:]])))
json.dump(out, open(sys.argv[2], "w"), default=float)
for r in rows: print(r)
for f, v in out["files"].items(): print(f, v["n"], v["true_k"], v["rec_k"], v["sil"], v["acc"], v["ari"], v["iters"], v["train_ms"], v["corr"], v["age"], v["rep"])
print(preds); print({kk: (np.mean(v), np.std(v), np.min(v)) for kk, v in out["stab"].items() if isinstance(v, list)}, out["stab"]["ari_pp"], out["stab"]["ari_r"])
print([(c["label"], c["customer_count"], c["avg_age"], c["avg_repeats"], c["top_product"], c["top_product_share"], c["top_district"], c["female_share"]) for c in out["files"]["tn_gift_shop_100.csv"]["clusters"]])

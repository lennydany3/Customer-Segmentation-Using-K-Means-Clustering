import sys, json, time
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'backend'))
import numpy as np, pandas as pd
from model import CustomerSegmentationModel
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
df = pd.read_csv(sys.argv[1])
print(len(df), df.describe().round(2).to_string())
m = CustomerSegmentationModel()
print(m.load_dataset(df))
ks, wcss, rk = m.compute_elbow()
X = m.get_features(); Xs = StandardScaler().fit_transform(X)
sils=[round(silhouette_score(Xs, KMeans(k,random_state=42,n_init=10).fit_predict(Xs)),4) for k in ks]
dbs=[round(davies_bouldin_score(Xs, KMeans(k,random_state=42,n_init=10).fit_predict(Xs)),4) for k in ks]
chs=[round(calinski_harabasz_score(Xs, KMeans(k,random_state=42,n_init=10).fit_predict(Xs)),2) for k in ks]
print('k',ks,'\nwcss',wcss,'\nsil',sils,'\ndb',dbs,'\nch',chs,'\nrec',rk)
t=time.time(); res = m.train(rk); print('train time', time.time()-t)
print('sil', res['silhouette_score'])
for c in res['clusters']: print(c['cluster_id'], c['label'], c['customer_count'], c['percentage'], c['avg_age'], c['avg_income'], c['avg_spending'], c['centroid'])
print('iters', m.model.n_iter_, 'inertia', m.model.inertia_)
for q in [(95,85),(90,15),(25,80),(25,20),(55,50)]:
    p=m.predict(30,*q); print(q, p['cluster_id'], p['label'])
k=rk
def ev(name, lab):
    lab=np.asarray(lab); n=len(set(lab))
    if n<2: return dict(model=name,k=n,sil=None,db=None,ch=None)
    return dict(model=name,k=n,sil=round(silhouette_score(Xs,lab),3),db=round(davies_bouldin_score(Xs,lab),3),ch=round(calinski_harabasz_score(Xs,lab),1))
rng=np.random.default_rng(42)
rows=[]
rows.append(ev('KMeans proposed (scaled, k-means++, n_init=10)', m.model.labels_))
rows.append(ev('Agglomerative (Ward)', AgglomerativeClustering(k,linkage='ward').fit_predict(Xs)))
rows.append(ev('Gaussian Mixture', GaussianMixture(k,random_state=42).fit_predict(Xs)))
rows.append(ev('BIRCH', Birch(n_clusters=k).fit_predict(Xs)))
rows.append(ev('MiniBatch KMeans', MiniBatchKMeans(k,random_state=0,n_init=1,batch_size=20).fit_predict(Xs)))
rows.append(ev('KMeans random init, 1 run (mean of 20 seeds)', None) if False else None)
s=[];d=[];c=[]
for seed in range(20):
    l=KMeans(k,init='random',n_init=1,random_state=seed).fit_predict(Xs); s.append(silhouette_score(Xs,l)); d.append(davies_bouldin_score(Xs,l)); c.append(calinski_harabasz_score(Xs,l))
rows[-1]=dict(model='KMeans random init, single run (avg 20 seeds)',k=k,sil=round(np.mean(s),3),db=round(np.mean(d),3),ch=round(np.mean(c),1),silmin=round(min(s),3))
rows.append(ev('Agglomerative (average)', AgglomerativeClustering(k,linkage='average').fit_predict(Xs)))
rows.append(ev('Agglomerative (complete)', AgglomerativeClustering(k,linkage='complete').fit_predict(Xs)))
rows.append(ev('Agglomerative (single)', AgglomerativeClustering(k,linkage='single').fit_predict(Xs)))
rows.append(ev('DBSCAN (eps=0.5)', DBSCAN(eps=0.5,min_samples=5).fit_predict(Xs)))
inc=df['Annual Income (k$)'].values
rows.append(ev('Rule-based income tiers', pd.qcut(inc,k,labels=False,duplicates='drop')))
rows.append(ev('Income-only KMeans', KMeans(k,random_state=42,n_init=10).fit_predict(Xs[:,[0]])))
rr=[ev('Random assignment', rng.integers(0,k,len(Xs))) for _ in range(20)]
rows.append(dict(model='Random assignment (avg 20)',k=k,sil=round(np.mean([r['sil'] for r in rr]),3),db=round(np.mean([r['db'] for r in rr]),3),ch=round(np.mean([r['ch'] for r in rr]),1)))
for r in rows: print(r)
# stability: ARI across seeds
from sklearn.metrics import adjusted_rand_score
labs=[KMeans(k,random_state=s,n_init=10).fit_predict(Xs) for s in range(10)]
print('ARI kmeans++ n_init10 across seeds', np.mean([adjusted_rand_score(labs[0],l) for l in labs[1:]]))
labs=[KMeans(k,init='random',random_state=s,n_init=1).fit_predict(Xs) for s in range(10)]
print('ARI random 1 run', np.mean([adjusted_rand_score(labs[0],l) for l in labs[1:]]))
# unscaled
print('unscaled sil', silhouette_score(Xs, KMeans(k,random_state=42,n_init=10).fit_predict(X)))
json.dump(dict(ks=ks,wcss=wcss,sil=sils,db=dbs,ch=chs,rk=rk,rows=rows,clusters=res['clusters'],points=res['points']), open(sys.argv[2],'w'), default=float)

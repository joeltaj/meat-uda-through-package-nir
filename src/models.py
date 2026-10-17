"""Domain adaptation models and fillet-level evaluation with fixed parameters.

No parameter search is performed by this module.
"""

from pathlib import Path

from collections import defaultdict

import argparse, csv, time, math, warnings

import numpy as np

import pandas as pd

import scipy, scipy.linalg

from scipy.signal import savgol_filter

from scipy.stats import t as student_t

from sklearn.decomposition import PCA

from sklearn.metrics import f1_score

from sklearn.metrics.pairwise import rbf_kernel

RANDOM_STATE = 42

N_ESTIMATORS = 30

MAX_FEATURES = 60

TEST_FRACTION = 0.30

REGULARIZATION = 1e-8

DEFAULT_N_SPLITS = 25

UDA_FRACS = np.array([0.0, 0.05, 0.10, 0.20, 0.50, 0.80, 1.00], float)

TARGET_MODES = ("OM", "TP", "TB")

MODE_INDEX = {"OM": 0, "TP": 1, "TB": 2}

MODEL_CONFIG = {
    "RSDE": {"n_estimators": 30, "max_features_raw": 60, "regularization": 1e-8},
    "CORAL": {"reg": 1e-3},
    "SA": {"var": 0.95},
    "TCA": {"n_components": 50, "mu": 0.1, "gamma": "1/n_features"},
    "GFK": {"var": 0.95},
    "SNV": {"epsilon": 1e-10},
    "PseudoLabel": {"n_iter": 2, "conf_thresh": 0.80},
    "KMM": {"B": 5.0, "n_ref": "min(200,n_source)"},
    "DS": {"reg": 1e-3},
    "TrAdaBoost": {"T": 15, "conf_thresh": 0.75},
    "H1_GFK_DTCA": {"n_components": 50, "mu_reg": 0.1, "alpha": 1.0, "var": 0.95},
    "H2": {"adapted_experts": ["GFK", "CORAL"], "scale": "median positive squared distance"},
}

BASELINE_METHODS = [
    "CORAL", "SA", "TCA", "GFK",
    "SNV+CORAL", "SNV+SA", "SNV+TCA", "SNV+GFK",
    "H1-GFK-DTCA", "Pseudo-Label", "KMM", "DS", "TrAdaBoost",
]

ABLATION_METHODS = [
    "Expert-Source", "Expert-GFK", "Expert-CORAL",
    "Ensemble-EqualAvg", "Ensemble-MajorityVote", "Ensemble-ConfidenceWeighted",
    "H2",
]

SENSITIVITY_METHODS = []

ALL_METHODS = BASELINE_METHODS + ABLATION_METHODS

def load_data(path: Path):
    df = pd.read_csv(path)
    df["Sample_number"] = df["Sample_number"].astype(str).str.strip()
    df["Scan_type"] = df["Scan_type"].astype(str).str.strip().str.upper()
    df["Freshness"] = df["Freshness"].astype(str).str.strip().str.upper()
    xcols = list(df.columns[4:])
    X = df[xcols].to_numpy(float)
    sid = df["Sample_number"].to_numpy(str)
    mode = df["Scan_type"].to_numpy(str)
    y = df["Freshness"].to_numpy(str)
    return df, X, sid, mode, y, xcols

class Preprocessor:
    def __init__(self, kind, X_raw, train_mask):
        self.kind = kind
        if kind == "raw":
            self.mu = self.sd = None
        elif kind == "sgz":
            base = savgol_filter(X_raw, window_length=11, polyorder=2, deriv=1, axis=1)
            self.mu = base[train_mask].mean(0)
            self.sd = base[train_mask].std(0) + 1e-8
        else:
            raise ValueError(kind)
    def __call__(self, A):
        if self.kind == "raw": return np.asarray(A, float)
        if len(A) == 0: return np.asarray(A, float)
        B = savgol_filter(np.asarray(A,float), window_length=11, polyorder=2, deriv=1, axis=1)
        return (B - self.mu) / self.sd

class RSDE:
    def __init__(self, n_estimators=30, max_features=60, random_state=42, regularization=1e-8):
        self.n_estimators=n_estimators; self.max_features=max_features
        self.random_state=random_state; self.regularization=regularization
    def fit(self, X, y):
        X=np.asarray(X,float); y=np.asarray(y)
        self.classes_=np.unique(y)
        if len(self.classes_) != 2: raise ValueError("RSDE is binary only")
        m1 = y == self.classes_[1]
        X0,X1=X[~m1],X[m1]
        mu0,mu1=X0.mean(0),X1.mean(0)
        C0,C1=X0-mu0,X1-mu1
        pooled=(C0.T@C0 + C1.T@C1)/max(len(X)-2,1)
        prior0,prior1=len(X0)/len(X),len(X1)/len(X)
        rng=np.random.default_rng(self.random_state)
        m=min(self.max_features,X.shape[1]); self.models_=[]
        for _ in range(self.n_estimators):
            idx=np.sort(rng.choice(X.shape[1],m,replace=False))
            cov=pooled[np.ix_(idx,idx)].copy()
            ridge=self.regularization*(np.trace(cov)/m if np.trace(cov)>0 else 1)
            cov.flat[::m+1]+=ridge
            delta=mu1[idx]-mu0[idx]
            try: w=np.linalg.solve(cov,delta)
            except np.linalg.LinAlgError: w=np.linalg.pinv(cov)@delta
            b=-0.5*(mu1[idx]+mu0[idx])@w + np.log((prior1+1e-15)/(prior0+1e-15))
            self.models_.append((idx,w,b))
        return self
    def predict_proba(self,X):
        X=np.asarray(X,float); p1=np.zeros(len(X))
        for idx,w,b in self.models_:
            z=np.clip(X[:,idx]@w+b,-50,50); p1 += 1/(1+np.exp(-z))
        p1/=len(self.models_)
        return np.column_stack([1-p1,p1])

class WeightedRSDE(RSDE):
    def fit(self,X,y,sample_weight=None):
        if sample_weight is None: return super().fit(X,y)
        X=np.asarray(X,float); y=np.asarray(y); sw=np.asarray(sample_weight,float)
        sw=sw*len(y)/sw.sum(); self.classes_=np.unique(y)
        m1=y==self.classes_[1]; X0,X1=X[~m1],X[m1]; w0,w1=sw[~m1],sw[m1]
        mu0=np.average(X0,weights=w0,axis=0); mu1=np.average(X1,weights=w1,axis=0)
        C0,C1=X0-mu0,X1-mu1
        pooled=((w0[:,None]*C0).T@C0 + (w1[:,None]*C1).T@C1)/max(sw.sum()-2,1)
        prior0,prior1=w0.sum()/sw.sum(),w1.sum()/sw.sum()
        rng=np.random.default_rng(self.random_state); m=min(self.max_features,X.shape[1]); self.models_=[]
        for _ in range(self.n_estimators):
            idx=np.sort(rng.choice(X.shape[1],m,replace=False))
            cov=pooled[np.ix_(idx,idx)].copy(); ridge=self.regularization*(np.trace(cov)/m if np.trace(cov)>0 else 1)
            cov.flat[::m+1]+=ridge; delta=mu1[idx]-mu0[idx]
            try: w=np.linalg.solve(cov,delta)
            except np.linalg.LinAlgError: w=np.linalg.pinv(cov)@delta
            b=-0.5*(mu1[idx]+mu0[idx])@w + np.log((prior1+1e-15)/(prior0+1e-15))
            self.models_.append((idx,w,b))
        return self

def rsde_for(nfeat, raw_nfeat):
    mf=max(2,round(nfeat*MAX_FEATURES/raw_nfeat))
    return RSDE(N_ESTIMATORS,mf,RANDOM_STATE,REGULARIZATION)

def wrsde_for(nfeat, raw_nfeat):
    mf=max(2,round(nfeat*MAX_FEATURES/raw_nfeat))
    return WeightedRSDE(N_ESTIMATORS,mf,RANDOM_STATE,REGULARIZATION)

def _pca_basis(Xc,var=0.95):
    nmax=max(2,min(Xc.shape[0]-1,Xc.shape[1]))
    pca=PCA(n_components=nmax).fit(Xc)
    cum=np.cumsum(pca.explained_variance_ratio_); d=max(2,int(np.searchsorted(cum,var))+1)
    return pca.components_[:min(d,nmax)].T

def _sqrtm_psd(A):
    ev,evec=np.linalg.eigh(A); return evec@np.diag(np.sqrt(np.maximum(ev,0)))@evec.T

def _invsqrtm_psd(A):
    ev,evec=np.linalg.eigh(A); return evec@np.diag(1/np.sqrt(np.maximum(ev,1e-12)))@evec.T

def snv(A):
    A=np.asarray(A,float); return (A-A.mean(1,keepdims=True))/(A.std(1,keepdims=True)+1e-10)

def coral(Xs,Xu,Xt,reg=1e-3):
    p=Xs.shape[1]; mus,mut=Xs.mean(0),Xu.mean(0)
    Cs=np.cov(Xs.T)+reg*np.eye(p); Ct=np.cov(Xu.T)+reg*np.eye(p)
    W=_invsqrtm_psd(Cs)@_sqrtm_psd(Ct)
    return (Xs-mus)@W+mut, Xt

def sa(Xs,Xu,Xt,var=0.95):
    mus,mut=Xs.mean(0),Xu.mean(0); Vs=_pca_basis(Xs-mus,var); Vt=_pca_basis(Xu-mut,var)
    d=min(Vs.shape[1],Vt.shape[1]); Vs,Vt=Vs[:,:d],Vt[:,:d]
    return (Xs-mus)@(Vs@Vs.T@Vt), (Xt-mut)@Vt

def tca(Xs,Xu,Xt,n_comp=50,mu=0.1):
    ns,nt=len(Xs),len(Xu); n=ns+nt
    Xall=np.vstack([Xs,Xu]); mn,sd=Xall.mean(0),Xall.std(0)+1e-8
    Xc=np.vstack([(Xs-mn)/sd,(Xu-mn)/sd]); Xt_n=(Xt-mn)/sd
    g=1/Xs.shape[1]; K=rbf_kernel(Xc,gamma=g)+1e-5*np.eye(n)
    L=np.full((n,n),-1/(ns*nt)); L[:ns,:ns]=1/ns**2; L[ns:,ns:]=1/nt**2
    H=np.eye(n)-np.ones((n,n))/n; k=min(n_comp,n-2)
    try:
        _,W=scipy.linalg.eigh(K@H@K,K@L@K+mu*np.eye(n),subset_by_index=[n-k,n-1]); W=np.real(W[:,::-1])
    except Exception:
        ev,W=np.linalg.eigh(np.linalg.solve(K@L@K+mu*np.eye(n),K@H@K)); W=np.real(W[:,np.argsort(ev)[::-1][:k]])
    return K[:ns]@W, rbf_kernel(Xt_n,Xc,gamma=g)@W

def _gfk_full(Xs,Xu,Xt,var=0.95):
    p=Xs.shape[1]; mus,mut=Xs.mean(0),Xu.mean(0)
    Ps=_pca_basis(Xs-mus,var); Pt=_pca_basis(Xu-mut,var); d=min(Ps.shape[1],Pt.shape[1]); Ps,Pt=Ps[:,:d],Pt[:,:d]
    _,_,Vh=np.linalg.svd(Ps.T,full_matrices=True); Pp=Vh[d:].T
    U,sig,VtM=np.linalg.svd(Ps.T@Pt,full_matrices=False); theta=np.arccos(np.clip(sig,-1,1))
    Cu,_,_=np.linalg.svd(Pp.T@Pt@VtM.T,full_matrices=False); Ub,Gb=Ps@U,Pp@Cu[:,:d]; G=np.eye(p)
    for i,th in enumerate(theta):
        l1=2.0 if th<1e-10 else 1+np.sin(2*th)/(2*th); l2=0.0 if th<1e-10 else 1-np.sin(2*th)/(2*th)
        u,g=Ub[:,i:i+1],Gb[:,i:i+1]; G+=(l1-1)*(u@u.T)+(l2-1)*(g@g.T)
    Gh=_sqrtm_psd(G)
    return (Xs-mus)@Gh,(Xu-mut)@Gh,(Xt-mut)@Gh

def gfk(Xs,Xu,Xt,var=0.95):
    a,_,b=_gfk_full(Xs,Xu,Xt,var); return a,b

def hybrid1(Xs,Xu,Xt,ys,n_comp=50,mu_reg=0.1,alpha=1.0,var=0.95):
    Xsg,Xug,Xtg=_gfk_full(Xs,Xu,Xt,var); ns,nt=len(Xsg),len(Xug); n=ns+nt
    Xall=np.vstack([Xsg,Xug]); mn,sd=Xall.mean(0),Xall.std(0)+1e-8; Xc=(Xall-mn)/sd; Xt_n=(Xtg-mn)/sd
    g=1/Xsg.shape[1]; K=rbf_kernel(Xc,gamma=g)+1e-5*np.eye(n)
    L=np.full((n,n),-1/(ns*nt)); L[:ns,:ns]=1/ns**2; L[ns:,ns:]=1/nt**2
    H=np.eye(n)-np.ones((n,n))/n; cls=np.unique(ys); b=np.zeros(ns)
    b[ys==cls[0]]=1/max((ys==cls[0]).sum(),1); b[ys==cls[1]]=-1/max((ys==cls[1]).sum(),1)
    Hb=np.zeros((n,n)); Hb[:ns,:ns]=np.outer(b,b)
    RHS=K@(H+alpha*Hb)@K; LHS=K@L@K+mu_reg*np.eye(n); k=min(n_comp,n-2)
    try:
        _,W=scipy.linalg.eigh(RHS,LHS,subset_by_index=[n-k,n-1]); W=np.real(W[:,::-1])
    except Exception:
        ev,W=np.linalg.eigh(np.linalg.solve(LHS,RHS)); W=np.real(W[:,np.argsort(ev)[::-1][:k]])
    return K[:ns]@W, rbf_kernel(Xt_n,Xc,gamma=g)@W

def pseudo_label(Xs,ys,Xu,n_iter=2,conf_thresh=0.80,raw_nfeat=125):
    clf=rsde_for(Xs.shape[1],raw_nfeat); clf.fit(Xs,ys)
    for _ in range(n_iter):
        p=clf.predict_proba(Xu); mask=p.max(1)>=conf_thresh
        if mask.sum()<3: break
        yp=clf.classes_[np.argmax(p[mask],1)]; clf=rsde_for(Xs.shape[1],raw_nfeat)
        clf.fit(np.vstack([Xs,Xu[mask]]),np.concatenate([ys,yp]))
    return clf

def kmm_weights(Xs,Xu,B=5.0):
    nref=min(200,len(Xs)); d=np.sum((Xs[:nref,None]-Xu[None,:])**2,axis=2); sigma=np.median(d)+1e-8
    dfull=np.sum((Xs[:,None]-Xu[None,:])**2,axis=2); w=np.exp(-dfull/sigma).mean(1); w/=w.mean(); w=np.clip(w,1/B,B); return w/w.mean()

def direct_standardize(Xs,Xu,Xt,reg=1e-3):
    p=Xs.shape[1]; mus,mut=Xs.mean(0),Xu.mean(0); Cs=np.cov(Xs.T)+reg*np.eye(p); Ct=np.cov(Xu.T)+reg*np.eye(p)
    W=_invsqrtm_psd(Ct)@_sqrtm_psd(Cs); return Xs,(Xt-mut)@W+mus

def tradaboost_uda(Xs,ys,Xu,Xt,T=15,conf_thresh=0.75,raw_nfeat=125):
    ns=len(Xs); clf0=rsde_for(Xs.shape[1],raw_nfeat); clf0.fit(Xs,ys); p0=clf0.predict_proba(Xu); mask=p0.max(1)>=conf_thresh
    if mask.sum()<3: return clf0.classes_,clf0.predict_proba(Xt)
    Xuc=Xu[mask]; yuc=clf0.classes_[np.argmax(p0[mask],1)]; nt=len(Xuc); Xall=np.vstack([Xs,Xuc]); yall=np.concatenate([ys,yuc])
    w=np.concatenate([np.ones(ns)/ns,np.ones(nt)/nt]); beta_s=1/(1+np.sqrt(2*np.log(ns)/T)); clfs=[]; betas=[]
    for _ in range(T):
        wn=w/w.sum(); clf=wrsde_for(Xall.shape[1],raw_nfeat); clf.fit(Xall,yall,sample_weight=wn)
        predt=clf.classes_[np.argmax(clf.predict_proba(Xuc),1)]; et=(wn[ns:]*(predt!=yuc)).sum()/wn[ns:].sum(); et=np.clip(et,1e-10,0.5-1e-10); bt=et/(1-et)
        preds=clf.classes_[np.argmax(clf.predict_proba(Xs),1)]; w[:ns]*=beta_s**(preds==ys); w[ns:]*=bt**(1-(predt!=yuc))
        clfs.append(clf); betas.append(bt)
    T2=max(1,T//2); lw=np.maximum(0,[np.log(1/max(b,1e-10)) for b in betas[T2:]])
    if np.sum(lw)==0: lw=np.ones(T-T2)
    final=sum(lw[i]*clfs[T2+i].predict_proba(Xt) for i in range(T-T2))/np.sum(lw)
    return clf0.classes_,final

def fillet_probability_f1(y_scan, classes, proba, sids):
    """Mean predicted probability within each (fillet, condition), then macro-F1."""
    buckets=defaultdict(list)
    for sid,yt,p in zip(sids,y_scan,proba): buckets[(sid,yt)].append(p)
    yt_u=[]; yp_u=[]
    for (sid,yt),arr in buckets.items():
        pm=np.mean(arr,axis=0); yt_u.append(yt); yp_u.append(classes[int(np.argmax(pm))])
    return f1_score(yt_u,yp_u,average="macro",labels=["FR","TH"]), len(yt_u)

def equal_avg(p1,p2,p3): return (p1+p2+p3)/3

def majority_vote_probs(p1,p2,p3):
    pred=np.stack([np.argmax(p1,1),np.argmax(p2,1),np.argmax(p3,1)],axis=1)
    maj=(pred.sum(1)>=2).astype(int)
    out=np.zeros_like(p1); out[np.arange(len(maj)),maj]=1.0; return out

def confidence_weighted(p1,p2,p3):
    P=np.stack([p1,p2,p3],axis=0)
    ent=-np.sum(P*np.log(np.clip(P,1e-12,1)),axis=2)/np.log(P.shape[2])
    conf=np.clip(1-ent,1e-6,None); w=conf/conf.sum(0,keepdims=True)
    return np.sum(w[:,:,None]*P,axis=0)

def h2_probabilities(Xs,Xu,Xt,p_src,p_gfk,p_coral):
    """Information-equivalent 2-domain gate; no third domain and no test-derived scale."""
    mus,mut=Xs.mean(0),Xu.mean(0)
    dss=np.sum((Xs-mus)**2,1); dst=np.sum((Xs-mut)**2,1)
    dts=np.sum((Xu-mus)**2,1); dtt=np.sum((Xu-mut)**2,1)
    ref=np.concatenate([dss,dst,dts,dtt]); pos=ref[ref>0]; sigma=(np.median(pos) if len(pos) else 1.0)+1e-8
    ds=np.sum((Xt-mus)**2,1); dt=np.sum((Xt-mut)**2,1)
    logits=np.vstack([-ds/sigma,-dt/sigma]); logits-=logits.max(0,keepdims=True); w=np.exp(logits); w/=w.sum(0,keepdims=True)
    p_adapt=(p_gfk+p_coral)/2
    return w[0,:,None]*p_src + w[1,:,None]*p_adapt

def evaluate_one_split(split_seed, X_raw, sample_ids, scan_type, freshness, preprocess="raw", include_slow=True):
    unique=np.array(sorted(np.unique(sample_ids))); ntest=round(TEST_FRACTION*len(unique)); rng=np.random.default_rng(split_seed)
    perm=rng.permutation(unique); test_ids=set(perm[:ntest]); train_ids=set(perm[ntest:])
    mode_mask=lambda m,ids: (scan_type==m)&np.isin(sample_ids,list(ids))
    prep=Preprocessor(preprocess,X_raw,mode_mask("OM",train_ids))
    def data(m,ids):
        mask=mode_mask(m,ids); return prep(X_raw[mask]),freshness[mask],sample_ids[mask]
    Xs,ys,_=data("OM",train_ids); raw_nfeat=X_raw.shape[1]
    source=rsde_for(Xs.shape[1],raw_nfeat).fit(Xs,ys)
    rows=[]
    for mode_i,mode in enumerate(TARGET_MODES):
        Xu_pool,yu_pool,sid_pool=data(mode,train_ids); Xt,yt,sid_t=data(mode,test_ids)
        # references
        floor_p=source.predict_proba(Xt); floor,n_test_units=fillet_probability_f1(yt,source.classes_,floor_p,sid_t)
        # Target TRAIN labels are used only for these explicitly supervised reference conditions.
        # They are never passed to any unsupervised adaptation method or to H2.
        pooled=rsde_for(Xs.shape[1],raw_nfeat).fit(np.vstack([Xs,Xu_pool]),np.concatenate([ys,yu_pool]))
        pooled_f1,_=fillet_probability_f1(yt,pooled.classes_,pooled.predict_proba(Xt),sid_t)
        ind=rsde_for(Xu_pool.shape[1],raw_nfeat).fit(Xu_pool,yu_pool); ind_f1,_=fillet_probability_f1(yt,ind.classes_,ind.predict_proba(Xt),sid_t)
        for ref,val in [("source_only",floor),("in_domain_supervised",ind_f1),("supervised_pooled_reference",pooled_f1)]:
            rows.append(dict(split=split_seed,mode=mode,frac=0.0,method=ref,f1=val,n_target_fillets=np.nan,n_target_spectra=np.nan,n_test_units=n_test_units,kind="reference"))
        # one permutation of eligible target TRAIN fillets -> nested fractions
        eligible=np.array(sorted(np.unique(sid_pool))); rr=np.random.default_rng(100000+split_seed*17+mode_i); order=rr.permutation(eligible)
        for frac in UDA_FRACS:
            if frac==0:
                for meth in ALL_METHODS:
                    rows.append(dict(split=split_seed,mode=mode,frac=0.0,method=meth,f1=floor,n_target_fillets=0,n_target_spectra=0,n_test_units=n_test_units,kind="method"))
                continue
            n=max(1,int(math.ceil(frac*len(order)))); sel=set(order[:n]); mask=np.isin(sid_pool,list(sel)); Xu=Xu_pool[mask]
            nt_spectra=len(Xu)
            # common three experts used by GFK/CORAL and all H2 ablations
            p_src=floor_p
            Xsg,_,Xtg=_gfk_full(Xs,Xu,Xt); eg=rsde_for(Xsg.shape[1],raw_nfeat).fit(Xsg,ys); p_gfk=eg.predict_proba(Xtg)
            Xsc,Xtc=coral(Xs,Xu,Xt); ec=rsde_for(Xsc.shape[1],raw_nfeat).fit(Xsc,ys); p_coral=ec.predict_proba(Xtc)
            probs={
                "Expert-Source":p_src,"Expert-GFK":p_gfk,"Expert-CORAL":p_coral,
                "Ensemble-EqualAvg":equal_avg(p_src,p_gfk,p_coral),
                "Ensemble-MajorityVote":majority_vote_probs(p_src,p_gfk,p_coral),
                "Ensemble-ConfidenceWeighted":confidence_weighted(p_src,p_gfk,p_coral),
                "H2":h2_probabilities(Xs,Xu,Xt,p_src,p_gfk,p_coral),
                "CORAL":p_coral,"GFK":p_gfk,
            }
            # remaining baselines
            try:
                Xsa,Xta=sa(Xs,Xu,Xt); c=rsde_for(Xsa.shape[1],raw_nfeat).fit(Xsa,ys); probs["SA"]=c.predict_proba(Xta)
            except Exception: probs["SA"]=None
            if include_slow:
                try:
                    Xst,Xtt=tca(Xs,Xu,Xt); c=rsde_for(Xst.shape[1],raw_nfeat).fit(Xst,ys); probs["TCA"]=c.predict_proba(Xtt)
                except Exception: probs["TCA"]=None
                try:
                    Zs,Zt=hybrid1(Xs,Xu,Xt,ys); c=rsde_for(Zs.shape[1],raw_nfeat).fit(Zs,ys); probs["H1-GFK-DTCA"]=c.predict_proba(Zt)
                except Exception: probs["H1-GFK-DTCA"]=None
            else:
                probs["TCA"]=None; probs["H1-GFK-DTCA"]=None
            for base,fn in [("CORAL",coral),("SA",sa),("TCA",tca),("GFK",gfk)]:
                name="SNV+"+base
                if (not include_slow) and base=="TCA": probs[name]=None; continue
                try:
                    Xa,Xb=fn(snv(Xs),snv(Xu),snv(Xt)); c=rsde_for(Xa.shape[1],raw_nfeat).fit(Xa,ys); probs[name]=c.predict_proba(Xb)
                except Exception: probs[name]=None
            try:
                c=pseudo_label(Xs,ys,Xu,raw_nfeat=raw_nfeat); probs["Pseudo-Label"]=c.predict_proba(Xt)
            except Exception: probs["Pseudo-Label"]=None
            try:
                w=kmm_weights(Xs,Xu); c=wrsde_for(Xs.shape[1],raw_nfeat).fit(Xs,ys,sample_weight=w); probs["KMM"]=c.predict_proba(Xt)
            except Exception: probs["KMM"]=None
            try:
                Xa,Xb=direct_standardize(Xs,Xu,Xt); c=rsde_for(Xa.shape[1],raw_nfeat).fit(Xa,ys); probs["DS"]=c.predict_proba(Xb)
            except Exception: probs["DS"]=None
            try:
                cls,p=tradaboost_uda(Xs,ys,Xu,Xt,raw_nfeat=raw_nfeat); probs["TrAdaBoost"]=p
            except Exception: probs["TrAdaBoost"]=None
            for meth in ALL_METHODS:
                p=probs.get(meth)
                if p is None: val=np.nan
                else:
                    # All methods use the same class ordering because source labels are FR/TH.
                    val,_=fillet_probability_f1(yt,source.classes_,p,sid_t)
                rows.append(dict(split=split_seed,mode=mode,frac=float(frac),method=meth,f1=val,n_target_fillets=n,n_target_spectra=nt_spectra,n_test_units=n_test_units,kind="method"))
    return rows

def corrected_mean_ci(vals,test_fraction=TEST_FRACTION,alpha=0.05):
    vals=np.asarray(vals,float); vals=vals[np.isfinite(vals)]; n=len(vals)
    if n<2: return np.nan,np.nan,np.nan
    m=vals.mean(); s2=vals.var(ddof=1); train_fraction=1-test_fraction
    se=np.sqrt((1/n + test_fraction/train_fraction)*s2)
    crit=student_t.ppf(1-alpha/2,n-1); return m,m-crit*se,m+crit*se

def corrected_diff_test(a,b,test_fraction=TEST_FRACTION):
    a=np.asarray(a,float); b=np.asarray(b,float); mask=np.isfinite(a)&np.isfinite(b); d=a[mask]-b[mask]; n=len(d)
    if n<2 or np.allclose(d,d[0]): return (d.mean() if n else np.nan),np.nan,np.nan,np.nan
    md=d.mean(); s2=d.var(ddof=1); se=np.sqrt((1/n + test_fraction/(1-test_fraction))*s2); t=md/se if se>0 else np.inf
    p=2*student_t.sf(abs(t),n-1); crit=student_t.ppf(0.975,n-1); return md,md-crit*se,md+crit*se,p

def holm_adjust(pvals):
    p=np.asarray(pvals,float); out=np.full_like(p,np.nan); ok=np.where(np.isfinite(p))[0]
    if not len(ok): return out
    order=ok[np.argsort(p[ok])]; m=len(order); running=0
    for rank,idx in enumerate(order):
        adj=min(1.0,(m-rank)*p[idx]); running=max(running,adj); out[idx]=running
    return out

def make_summaries(res: pd.DataFrame, outdir: Path):
    methods=res[res.kind=="method"].copy()
    rec=[]
    for (mode,frac,meth),g in methods.groupby(["mode","frac","method"],sort=False):
        m,lo,hi=corrected_mean_ci(g.f1.values)
        rec.append(dict(mode=mode,frac=frac,method=meth,mean_f1=m,ci95_lo=lo,ci95_hi=hi,sd=np.nanstd(g.f1.values,ddof=1),n_splits=g.f1.notna().sum(),mean_target_fillets=np.nanmean(g.n_target_fillets),mean_target_spectra=np.nanmean(g.n_target_spectra)))
    pd.DataFrame(rec).to_csv(outdir/"summary_f1.csv",index=False)
    # references
    refs=res[res.kind=="reference"].copy(); rr=[]
    for (mode,meth),g in refs.groupby(["mode","method"]):
        m,lo,hi=corrected_mean_ci(g.f1.values); rr.append(dict(mode=mode,reference=meth,mean_f1=m,ci95_lo=lo,ci95_hi=hi,sd=np.nanstd(g.f1.values,ddof=1),n_splits=g.f1.notna().sum()))
    pd.DataFrame(rr).to_csv(outdir/"references.csv",index=False)
    # normalized AUC over fractions per split/mode/method
    auc_rows=[]
    for (split,mode,meth),g in methods.groupby(["split","mode","method"]):
        gg=g.sort_values("frac"); x=gg.frac.to_numpy(float); y=gg.f1.to_numpy(float)
        if len(x)==len(UDA_FRACS) and np.isfinite(y).all():
            auc=np.trapezoid(y,x)/(x.max()-x.min())
            auc_rows.append(dict(split=split,mode=mode,method=meth,auc_f1=auc))
    auc=pd.DataFrame(auc_rows); auc.to_csv(outdir/"auc_per_split.csv",index=False)
    # H2 vs pre-specified competitors on TP/TB AUC, corrected resampled t + Holm within domain
    competitors=["Expert-Source","Expert-GFK","Expert-CORAL","Ensemble-EqualAvg","Ensemble-MajorityVote","Ensemble-ConfidenceWeighted","CORAL","GFK","DS"]
    st=[]
    for mode in ["TP","TB"]:
        wide=auc[auc["mode"]==mode].pivot(index="split",columns="method",values="auc_f1")
        for comp in competitors:
            if "H2" not in wide or comp not in wide: continue
            md,lo,hi,p=corrected_diff_test(wide["H2"].values,wide[comp].values)
            st.append(dict(mode=mode,comparison=f"H2 - {comp}",mean_diff_auc=md,ci95_lo=lo,ci95_hi=hi,p_raw=p))
    st=pd.DataFrame(st)
    if len(st):
        st["p_holm"]=np.nan
        for mode,idx in st.groupby("mode").groups.items(): st.loc[idx,"p_holm"]=holm_adjust(st.loc[idx,"p_raw"].values)
    st.to_csv(outdir/"auc_corrected_stats.csv",index=False)

def validate_dataset(df, X, sample_ids, xcols):
    """Fail loudly if a different evaluation cohort is supplied."""
    if len(df) != 4053:
        raise ValueError(f"Expected 4,053 retained spectra; found {len(df)}")
    if len(np.unique(sample_ids)) != 153:
        raise ValueError(f"Expected 153 evaluation fillets; found {len(np.unique(sample_ids))}")
    if X.shape[1] != 125 or len(xcols) != 125:
        raise ValueError(f"Expected 125 wavelength variables; found {X.shape[1]}")

def outer_split_ids(sample_ids, split_seed):
    unique=np.array(sorted(np.unique(sample_ids)))
    ntest=round(TEST_FRACTION*len(unique))
    perm=np.random.default_rng(split_seed).permutation(unique)
    test_ids=set(perm[:ntest]); train_ids=set(perm[ntest:])
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == set(unique)
    return train_ids, test_ids

def target_order(eligible_ids, split_seed, mode):
    """Deterministic ordering for nested target adaptation subsets."""
    eligible=np.array(sorted(np.unique(eligible_ids)))
    rr=np.random.default_rng(100000 + split_seed*17 + MODE_INDEX[mode])
    return rr.permutation(eligible)

def make_split_manifest(sample_ids, n_splits=DEFAULT_N_SPLITS):
    rows=[]
    for seed in range(n_splits):
        tr,te=outer_split_ids(sample_ids,seed)
        rows += [dict(split=seed,Sample_number=x,outer_partition="train") for x in sorted(tr)]
        rows += [dict(split=seed,Sample_number=x,outer_partition="test") for x in sorted(te)]
    return pd.DataFrame(rows)

def leakage_audit(sample_ids, scan_type, n_splits=DEFAULT_N_SPLITS):
    """Check train/test isolation and nested target adaptation subsets."""
    rows=[]
    for seed in range(n_splits):
        tr,te=outer_split_ids(sample_ids,seed)
        for mode in TARGET_MODES:
            train_mode_ids=set(sample_ids[(scan_type==mode) & np.isin(sample_ids,list(tr))])
            test_mode_ids=set(sample_ids[(scan_type==mode) & np.isin(sample_ids,list(te))])
            overlap=train_mode_ids & test_mode_ids
            if overlap:
                raise AssertionError(f"Leakage in split {seed}, mode {mode}: {sorted(overlap)[:5]}")
            order=target_order(list(train_mode_ids),seed,mode)
            nested_prev=set()
            for frac in UDA_FRACS:
                if frac==0:
                    selected=set()
                else:
                    n=max(1,int(math.ceil(frac*len(order))))
                    selected=set(order[:n])
                if selected & test_mode_ids:
                    raise AssertionError(f"Target adaptation/test overlap in split {seed}, mode {mode}")
                if not nested_prev.issubset(selected):
                    raise AssertionError(f"Non-nested target subsets in split {seed}, mode {mode}")
                nested_prev=selected
            rows.append(dict(split=seed, mode=mode, n_outer_train=len(tr), n_outer_test=len(te),
                             n_mode_train_fillets=len(train_mode_ids), n_mode_test_fillets=len(test_mode_ids),
                             overlap_fillets=0, nested_target_subsets=True))
    return pd.DataFrame(rows)

def mmd2_rbf(A,B,gamma=1/125):
    A=np.asarray(A,float); B=np.asarray(B,float)
    return float(rbf_kernel(A,A,gamma=gamma).mean() + rbf_kernel(B,B,gamma=gamma).mean()
                 - 2*rbf_kernel(A,B,gamma=gamma).mean())

def dataset_accounting(df):
    g=(df.groupby(["Freshness","Scan_type"])
         .agg(n_spectra=("Sample_number","size"), n_fillets=("Sample_number","nunique"))
         .reset_index())
    return g

def mmd_reanalysis(df, X):
    rows=[]
    for mode in TARGET_MODES:
        a=X[(df.Scan_type.str.upper()==mode) & (df.Freshness.str.upper()=="FR")]
        b=X[(df.Scan_type.str.upper()==mode) & (df.Freshness.str.upper()=="TH")]
        rows.append(dict(comparison=f"FR-TH within {mode}",mmd2=mmd2_rbf(a,b)))
    for cond in ["FR","TH"]:
        om=X[(df.Scan_type.str.upper()=="OM") & (df.Freshness.str.upper()==cond)]
        for target in ["TP","TB"]:
            tt=X[(df.Scan_type.str.upper()==target) & (df.Freshness.str.upper()==cond)]
            rows.append(dict(comparison=f"OM-{target} {cond}",mmd2=mmd2_rbf(om,tt)))
    return pd.DataFrame(rows)

def evaluate_h2_split(split_seed, X_raw, sample_ids, scan_type, freshness):
    """Evaluate the information-equivalent H2 ablation for TP/TB."""
    train_ids,test_ids=outer_split_ids(sample_ids,split_seed)
    mode_mask=lambda m,ids: (scan_type==m)&np.isin(sample_ids,list(ids))
    prep=Preprocessor("raw",X_raw,mode_mask("OM",train_ids))
    def data(m,ids):
        mask=mode_mask(m,ids); return prep(X_raw[mask]),freshness[mask],sample_ids[mask]
    Xs,ys,_=data("OM",train_ids); raw_nfeat=X_raw.shape[1]
    source=rsde_for(Xs.shape[1],raw_nfeat).fit(Xs,ys)
    rows=[]
    methods=["Expert-Source","Expert-GFK","Expert-CORAL","Ensemble-EqualAvg",
             "Ensemble-MajorityVote","Ensemble-ConfidenceWeighted","H2"]
    for mode in ("TP","TB"):
        Xu_pool,_,sid_pool=data(mode,train_ids)
        Xt,yt,sid_t=data(mode,test_ids)
        p_src=source.predict_proba(Xt)
        floor,n_test_units=fillet_probability_f1(yt,source.classes_,p_src,sid_t)
        order=target_order(sid_pool,split_seed,mode)
        for frac in UDA_FRACS:
            if frac==0:
                for meth in methods:
                    rows.append(dict(split=split_seed,mode=mode,frac=0.0,method=meth,f1=floor,
                                     n_target_fillets=0,n_target_spectra=0,n_test_units=n_test_units))
                continue
            n=max(1,int(math.ceil(frac*len(order))))
            selected=set(order[:n]); mask=np.isin(sid_pool,list(selected)); Xu=Xu_pool[mask]
            # No target labels enter any of the following fits.
            Xsg,_,Xtg=_gfk_full(Xs,Xu,Xt,var=MODEL_CONFIG["GFK"]["var"])
            eg=rsde_for(Xsg.shape[1],raw_nfeat).fit(Xsg,ys); p_gfk=eg.predict_proba(Xtg)
            Xsc,Xtc=coral(Xs,Xu,Xt,reg=MODEL_CONFIG["CORAL"]["reg"])
            ec=rsde_for(Xsc.shape[1],raw_nfeat).fit(Xsc,ys); p_coral=ec.predict_proba(Xtc)
            probs={
                "Expert-Source":p_src,
                "Expert-GFK":p_gfk,
                "Expert-CORAL":p_coral,
                "Ensemble-EqualAvg":equal_avg(p_src,p_gfk,p_coral),
                "Ensemble-MajorityVote":majority_vote_probs(p_src,p_gfk,p_coral),
                "Ensemble-ConfidenceWeighted":confidence_weighted(p_src,p_gfk,p_coral),
                "H2":h2_probabilities(Xs,Xu,Xt,p_src,p_gfk,p_coral),
            }
            for meth,p in probs.items():
                val,_=fillet_probability_f1(yt,source.classes_,p,sid_t)
                rows.append(dict(split=split_seed,mode=mode,frac=float(frac),method=meth,f1=val,
                                 n_target_fillets=n,n_target_spectra=len(Xu),n_test_units=n_test_units))
    return rows

def ablation_auc_from_long(long_df):
    rows=[]
    for (split,mode,meth),g in long_df.groupby(["split","mode","method"]):
        g=g.sort_values("frac")
        x=g.frac.to_numpy(float); y=g.f1.to_numpy(float)
        if len(x)!=len(UDA_FRACS):
            raise ValueError(f"Incomplete curve: split={split}, mode={mode}, method={meth}")
        rows.append(dict(split=split,mode=mode,method=meth,
                         auc_f1=np.trapezoid(y,x)/(x.max()-x.min())))
    return pd.DataFrame(rows)

def ablation_summary(auc):
    name_map={
        "Expert-Source":"Source-only expert", "Expert-GFK":"GFK expert",
        "Expert-CORAL":"CORAL expert", "Ensemble-EqualAvg":"Equal probability average",
        "Ensemble-MajorityVote":"Majority vote", "Ensemble-ConfidenceWeighted":"Confidence weighting",
        "H2":"H2",
    }
    q=(auc.groupby(["method","mode"]).auc_f1.mean().unstack("mode")
          .reindex(list(name_map)).rename(index=name_map).reset_index()
          .rename(columns={"method":"Aggregation / expert","TP":"TP_AUC","TB":"TB_AUC"}))
    return q

def paired_ablation_statistics(auc):
    label={"Ensemble-EqualAvg":"Equal probability average",
           "Ensemble-MajorityVote":"Majority vote",
           "Ensemble-ConfidenceWeighted":"Confidence weighting"}
    rows=[]
    for mode in ["TP","TB"]:
        wide=auc[auc["mode"]==mode].pivot(index="split",columns="method",values="auc_f1")
        raw=[]
        for comp,pretty in label.items():
            a=wide["H2"].to_numpy(); b=wide[comp].to_numpy()
            md,lo,hi,p=corrected_diff_test(a,b)
            d=a-b; s=float(np.std(d,ddof=1)); n=len(d)
            se=float(np.sqrt((1/n + TEST_FRACTION/(1-TEST_FRACTION))*s*s))
            raw.append(dict(mode=mode,comparison=f"H2 - {pretty}",mean_diff_auc=md,
                            ci95_lo=lo,ci95_hi=hi,p_raw=p,n_splits=n,diff_sd=s,corrected_se=se))
        adj=holm_adjust([r["p_raw"] for r in raw])
        for r,padj in zip(raw,adj): r["p_holm"]=padj; rows.append(r)
    cols=["mode","comparison","mean_diff_auc","ci95_lo","ci95_hi","p_raw","p_holm","n_splits","diff_sd","corrected_se"]
    return pd.DataFrame(rows)[cols]

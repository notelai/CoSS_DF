from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA

class Standardizer:
    def __init__(self,floor=1e-12): self.floor=floor
    def fit(self,X):
        X=np.asarray(X,float); self.mean_=X.mean(0); self.sd_=np.maximum(X.std(0,ddof=0),self.floor); return self
    def transform(self,X): return (np.asarray(X,float)-self.mean_)/self.sd_
    def fit_transform(self,X): return self.fit(X).transform(X)

class SpatialPCAWhitener:
    def __init__(self,n_components,eig_floor=1e-12): self.n_components=n_components; self.eig_floor=eig_floor
    def fit(self,X):
        self.std=Standardizer().fit(X); Z=self.std.transform(X); self.pca=PCA(n_components=self.n_components,svd_solver="full",whiten=False).fit(Z)
        self.scale_=np.sqrt(np.maximum(self.pca.explained_variance_,self.eig_floor)); return self
    def transform(self,X): return self.pca.transform(self.std.transform(X))/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)

class RegularizedWhitener:
    def __init__(self,gamma=1e-6,floor=1e-12): self.gamma=gamma; self.floor=floor
    def fit(self,X):
        X=np.asarray(X,float); self.mean_=X.mean(0); Z=X-self.mean_; d=Z.shape[1]
        S=(Z.T@Z)/max(1,len(Z)-1); s=max(float(np.trace(S))/d,self.floor); A=S+self.gamma*s*np.eye(d)
        vals,vecs=np.linalg.eigh(A); vals=np.maximum(vals,self.floor); self.W_=(vecs*(1/np.sqrt(vals)))@vecs.T; self.cov_reg_=A; return self
    def transform(self,X): return (np.asarray(X,float)-self.mean_)@self.W_
    def fit_transform(self,X): return self.fit(X).transform(X)

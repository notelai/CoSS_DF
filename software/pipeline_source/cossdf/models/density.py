from __future__ import annotations
import numpy as np
from scipy.special import logsumexp
from scipy.linalg import solve_triangular

class IsotropicClassKDE:
    def __init__(self,h=1.0): self.h=float(h)
    def fit(self,X,y):
        self.classes_=np.unique(y); self.samples_={k:np.asarray(X[y==k],float) for k in self.classes_}; self.d_=X.shape[1]; return self
    def log_class_density(self,X):
        X=np.asarray(X,float); out=[]; h2=self.h*self.h
        const=-0.5*self.d_*np.log(2*np.pi)-self.d_*np.log(self.h)
        for k in self.classes_:
            T=self.samples_[k]; d2=((X[:,None,:]-T[None,:,:])**2).sum(2)
            out.append(const+logsumexp(-d2/(2*h2),axis=1)-np.log(len(T)))
        return np.column_stack(out)
    def predict_proba(self,X):
        L=self.log_class_density(X)-np.log(len(self.classes_)); return np.exp(L-logsumexp(L,axis=1,keepdims=True))

class GaussianClassDensity:
    def __init__(self,gamma=1e-6,floor=1e-12): self.gamma=gamma; self.floor=floor
    def fit(self,X,y):
        self.classes_=np.unique(y); self.params_={}; d=X.shape[1]
        for k in self.classes_:
            Z=np.asarray(X[y==k],float); mu=Z.mean(0); C=np.cov(Z,rowvar=False) if len(Z)>1 else np.eye(d)
            C=np.atleast_2d(C); s=max(float(np.trace(C))/d,self.floor); C=C+self.gamma*s*np.eye(d)
            vals=np.linalg.eigvalsh(C)
            if vals.min()<=0: C=C+(self.floor-vals.min()+self.floor)*np.eye(d)
            L=np.linalg.cholesky(C); logdet=2*np.log(np.diag(L)).sum(); self.params_[k]=(mu,L,logdet)
        self.d_=d; return self
    def predict_proba(self,X):
        X=np.asarray(X,float); logs=[]
        for k in self.classes_:
            mu,L,ld=self.params_[k]; q=solve_triangular(L,(X-mu).T,lower=True).T; logs.append(-0.5*(self.d_*np.log(2*np.pi)+ld+(q*q).sum(1)))
        A=np.column_stack(logs)-np.log(len(self.classes_)); return np.exp(A-logsumexp(A,axis=1,keepdims=True))

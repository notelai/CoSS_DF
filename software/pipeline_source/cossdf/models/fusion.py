from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

def nll(P,y):
    return float(-np.mean(np.log(P[np.arange(len(y)),y])))

def linear_pool(Ps,Pf,l): return l*Ps+(1-l)*Pf

def logop(Ps,Pf,l):
    L=l*np.log(Ps)+(1-l)*np.log(Pf); return np.exp(L-logsumexp(L,axis=1,keepdims=True))

def optimize_lambda(Ps,Pf,y,rule="logop"):
    fn=logop if rule=="logop" else linear_pool
    f=lambda l:nll(fn(Ps,Pf,float(l)),y)
    r=minimize_scalar(f,bounds=(0,1),method="bounded",options={"xatol":1e-12})
    candidates=[(0.0,f(0.0)),(1.0,f(1.0)),(float(r.x),float(r.fun))]
    candidates.sort(key=lambda x:(round(x[1],12),x[0]))
    return candidates[0][0],candidates[0][1]

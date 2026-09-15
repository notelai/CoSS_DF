from __future__ import annotations
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from ..preprocessing import load_preprocessed

LBP_SPECS=[(8,1),(16,2),(24,3)]

def lbp_hist_from_path(path,P,R):
    _,y=load_preprocessed(path); img=np.clip(np.round(y*255),0,255).astype(np.uint8)
    z=local_binary_pattern(img,P,R,method="uniform")
    h,_=np.histogram(z,bins=np.arange(P+3),range=(0,P+2),density=False); h=h.astype(float); h/=max(h.sum(),1)
    return h

def extract_lbp_matrix(paths,P,R): return np.vstack([lbp_hist_from_path(p,P,R) for p in paths])

def fit_svm(X,y,kernel,C,gamma_scale=None):
    if kernel=="linear": return make_pipeline(StandardScaler(),SVC(C=C,kernel="linear",probability=False)).fit(X,y)
    d=X.shape[1]; gamma=(gamma_scale/d)
    return make_pipeline(StandardScaler(),SVC(C=C,kernel="rbf",gamma=gamma,probability=False)).fit(X,y)

from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

def ece_equal_frequency(P,y,n_bins=15):
    conf=P.max(1); pred=P.argmax(1); order=np.argsort(conf); bins=np.array_split(order,n_bins); out=0.0
    for b in bins:
        if len(b): out += len(b)/len(y)*abs(np.mean(pred[b]==y[b])-np.mean(conf[b]))
    return float(out)

def multiclass_brier(P,y):
    Y=np.eye(P.shape[1])[y]; return float(np.mean(np.sum((P-Y)**2,axis=1)))

def classification_metrics(y,pred):
    return {"accuracy":float(accuracy_score(y,pred)),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),"macro_f1":float(f1_score(y,pred,average="macro"))}

def probabilistic_metrics(y,P):
    return {"nll":float(-np.mean(np.log(P[np.arange(len(y)),y]))),"brier":multiclass_brier(P,y),"ece":ece_equal_frequency(P,y,15)}

def all_metrics(y,P=None,pred=None):
    if pred is None: pred=P.argmax(1)
    d=classification_metrics(y,pred)
    if P is not None: d.update(probabilistic_metrics(y,P))
    return d

import numpy as np
import pandas as pd
from cossdf.geometry_control import _linear_simplex_coords, _select_logreg
from cossdf.features.spectral import spectral_from_G


def test_linear_simplex_control_dimension():
    rng=np.random.default_rng(1)
    G=rng.random((10,8,8)); G[:,0,0]=0
    Z=_linear_simplex_coords(G,6)
    assert Z.shape==(10,5)
    assert np.isfinite(Z).all()


def test_logreg_selection_returns_candidate():
    rng=np.random.default_rng(2)
    X=rng.normal(size=(60,8)); y=np.repeat([0,1,2],20)
    df=pd.DataFrame({'class_id':y,'source_id':np.nan})
    chosen,table=_select_logreg(X,y,df,'fmd')
    assert chosen['C'] in {.01,.1,1,10,100}
    assert len(table)==5

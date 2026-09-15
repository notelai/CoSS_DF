import numpy as np
import pandas as pd
from cossdf.features.spectral import radial_sector_counts, directional_spectral_from_G
from cossdf.directional_fusion_controls import _diagnostics, _select_matched_geometry_lr


def test_directional_partition_has_no_empty_cells():
    expected={4:[3,4,6,8,14,16,5,7],6:[1,2,4,5,4,5,9,10,8,9,2,4],8:[1,1,2,3,3,4,3,4,7,8,7,8,4,5,1,2]}
    for R,c in expected.items():
        assert radial_sector_counts(8,R,2)==c
        assert sum(c)==63 and min(c)>0


def test_directional_composition_positive_and_closed():
    rng=np.random.default_rng(4); G=rng.random((20,8,8)); G[:,0,0]=0
    E,P,Z=directional_spectral_from_G(G,6,2,1e-6)
    assert E.shape==(20,12); assert P.shape==(20,12); assert Z.shape==(20,11)
    assert np.all(P>0); assert np.allclose(P.sum(1),1.0); assert np.isfinite(Z).all()


def test_synergy_diagnostics_identity_fusion():
    y=np.array([0,1,0,1]); Ps=np.array([[.9,.1],[.4,.6],[.6,.4],[.2,.8]])
    Pf=np.array([[.8,.2],[.7,.3],[.4,.6],[.1,.9]])
    d=_diagnostics(y,Ps,Pf,Ps,'identity')
    assert abs(d['mean_logloss_gain_vs_spatial'])<1e-12
    assert d['fusion_harm_rate_given_spatial_correct']==0.0


def test_symmetric_geometry_selector_returns_shared_pair():
    rng=np.random.default_rng(5); n=90; y=np.repeat([0,1,2],30)
    G=rng.random((n,8,8)); G[:,0,0]=0
    df=pd.DataFrame({'class_id':y,'source_id':np.nan})
    chosen,tab,_=_select_matched_geometry_lr(G,y,df,'fmd',3)
    assert chosen['R'] in {4,6,8}; assert chosen['C'] in {.01,.1,1.,10.,100.}
    assert len(tab)==15

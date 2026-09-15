import numpy as np
from cossdf.models.transforms import SpatialPCAWhitener, RegularizedWhitener
from cossdf.models.density import IsotropicClassKDE, GaussianClassDensity
from cossdf.models.fusion import logop

def test_small_dual_branch_smoke():
    rng=np.random.default_rng(10); n=90; y=np.repeat(np.arange(3),30)
    X=rng.normal(size=(n,20))+y[:,None]*.2; Z=rng.normal(size=(n,5))+y[:,None]*.15
    tr=np.r_[0:20,30:50,60:80]; te=np.r_[20:30,50:60,80:90]
    st=SpatialPCAWhitener(4).fit(X[tr]); A=st.transform(X[tr]); B=st.transform(X[te]); Ps=IsotropicClassKDE(.75).fit(A,y[tr]).predict_proba(B)
    ft=RegularizedWhitener().fit(Z[tr]); C=ft.transform(Z[tr]); D=ft.transform(Z[te]); Pf=IsotropicClassKDE(.75).fit(C,y[tr]).predict_proba(D)
    P=logop(Ps,Pf,.5)
    assert P.shape==(30,3) and np.allclose(P.sum(1),1)

import numpy as np
from scipy.fft import dctn
from cossdf.features.spectral import block_dct_coordinate_energy, band_counts, energy_to_bands, smooth_composition, helmert_basis, ilr
from cossdf.models.transforms import RegularizedWhitener
from cossdf.models.density import IsotropicClassKDE
from cossdf.models.fusion import logop, optimize_lambda, nll


def test_band_counts():
    assert band_counts(8,4)==[7,14,30,12]
    assert band_counts(8,6)==[3,9,9,19,17,6]
    assert band_counts(8,8)==[2,5,7,7,15,15,9,3]


def test_dct_centering_parseval():
    rng=np.random.default_rng(1); y=rng.normal(size=(224,224))
    b=y.reshape(28,8,28,8).transpose(0,2,1,3).reshape(-1,8,8)
    x=b-b.mean((1,2),keepdims=True); C=dctn(x,type=2,norm="ortho",axes=(1,2))
    assert np.max(np.abs(C[:,0,0])) < 1e-10
    assert np.allclose(np.sum(x*x,axis=(1,2)),np.sum(C*C,axis=(1,2)),rtol=1e-12,atol=1e-10)


def test_affine_composition_invariance():
    rng=np.random.default_rng(2); y=rng.uniform(size=(224,224)); G=block_dct_coordinate_energy(y); E=energy_to_bands(G,8); p=smooth_composition(E)
    yp=1.25*y-0.2; Gp=block_dct_coordinate_energy(yp); Ep=energy_to_bands(Gp,8); pp=smooth_composition(Ep)
    assert np.allclose(p,pp,atol=1e-12,rtol=1e-12)
    assert np.allclose(ilr(p),ilr(pp),atol=1e-11,rtol=1e-11)


def test_composition_positive_and_closed():
    for E in [np.zeros(6),np.array([0,1,0,2,0,3.],float)]:
        p=smooth_composition(E)
        assert np.all(p>0)
        assert abs(p.sum()-1)<1e-14


def test_whitening_basis_invariance_for_isotropic_kde():
    rng=np.random.default_rng(3); X=rng.normal(size=(80,5)); y=np.repeat([0,1],40); T=rng.normal(size=(10,5))
    Q,_=np.linalg.qr(rng.normal(size=(5,5)))
    w1=RegularizedWhitener().fit(X); A1=w1.transform(X); B1=w1.transform(T); P1=IsotropicClassKDE(.75).fit(A1,y).predict_proba(B1)
    X2=X@Q; T2=T@Q; w2=RegularizedWhitener().fit(X2); A2=w2.transform(X2); B2=w2.transform(T2); P2=IsotropicClassKDE(.75).fit(A2,y).predict_proba(B2)
    assert np.allclose(P1,P2,atol=1e-9,rtol=1e-9)


def test_logop_endpoints_and_optimized_nll():
    rng=np.random.default_rng(4); A=rng.uniform(.01,1,size=(50,4)); B=rng.uniform(.01,1,size=(50,4)); A/=A.sum(1,keepdims=True); B/=B.sum(1,keepdims=True); y=rng.integers(0,4,size=50)
    assert np.allclose(logop(A,B,1),A)
    assert np.allclose(logop(A,B,0),B)
    lam,sc=optimize_lambda(A,B,y,"logop")
    assert 0<=lam<=1
    assert sc <= min(nll(A,y),nll(B,y))+1e-10

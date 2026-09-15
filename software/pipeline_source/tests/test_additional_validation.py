import numpy as np
from cossdf.features.alternative_spectral import _haar2_level_energy_block,dwt_energy_descriptor,gabor_energy_descriptor,energies_to_ilr,clr_ilr_sanity


def test_haar_energy_parseval_centered_block():
    rng=np.random.default_rng(1); x=rng.normal(size=(8,8)); x=x-x.mean()
    e=_haar2_level_energy_block(x)
    assert e.shape==(7,)
    assert np.all(e>=0)
    assert np.isclose(e.sum(),np.sum(x*x),rtol=1e-10,atol=1e-10)


def test_dwt_gabor_descriptors_are_positive_energy_vectors():
    rng=np.random.default_rng(2); y=rng.normal(size=(224,224))
    d=dwt_energy_descriptor(y); g=gabor_energy_descriptor(y)
    assert d.shape==(7,) and g.shape==(8,)
    assert np.all(d>=0) and np.all(g>=0)
    assert d.sum()>0 and g.sum()>0


def test_clr_ilr_distance_equivalence():
    E=np.array([[1,2,3,4],[4,3,2,1],[1,1,1,1.]],float)
    P,Z=energies_to_ilr(E)
    s=clr_ilr_sanity(P,max_pairs=20)
    assert s['max_abs_distance_diff'] < 1e-10
    assert s['max_abs_clr_reconstruction_error'] < 1e-10
    assert s['clr_cov_rank'] <= 3

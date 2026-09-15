from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from ..preprocessing import load_preprocessed
from .spectral import smooth_composition, helmert_basis


def _blocks_centered(y: np.ndarray, m: int = 8):
    h,w=y.shape
    hh=(h//m)*m; ww=(w//m)*m
    z=y[:hh,:ww]
    blocks=z.reshape(hh//m,m,ww//m,m).transpose(0,2,1,3).reshape(-1,m,m)
    return blocks-blocks.mean(axis=(1,2),keepdims=True)


def _haar2_level_energy_block(block: np.ndarray):
    """Seven-subband 2-level orthonormal Haar energy for one centered 8x8 block.

    Output order: LH1, HL1, HH1, LH2, HL2, HH2, LL2.
    The transform is orthonormal, so the seven energies sum to the centered
    block energy up to floating-point error.
    """
    s=math.sqrt(2.0)
    def level(x):
        lo_r=(x[:,0::2]+x[:,1::2])/s
        hi_r=(x[:,0::2]-x[:,1::2])/s
        LL=(lo_r[0::2,:]+lo_r[1::2,:])/s
        LH=(lo_r[0::2,:]-lo_r[1::2,:])/s
        HL=(hi_r[0::2,:]+hi_r[1::2,:])/s
        HH=(hi_r[0::2,:]-hi_r[1::2,:])/s
        return LL,LH,HL,HH
    LL1,LH1,HL1,HH1=level(block)
    LL2,LH2,HL2,HH2=level(LL1)
    return np.array([
        np.sum(LH1*LH1),np.sum(HL1*HL1),np.sum(HH1*HH1),
        np.sum(LH2*LH2),np.sum(HL2*HL2),np.sum(HH2*HH2),np.sum(LL2*LL2)
    ],dtype=np.float64)


def dwt_energy_descriptor(y: np.ndarray, m: int = 8):
    blocks=_blocks_centered(y,m)
    return np.mean(np.vstack([_haar2_level_energy_block(b) for b in blocks]),axis=0)


def _gabor_atoms(m: int = 8):
    """Fixed quadrature local Gabor bank.

    Fixed Gabor bank: four orientations x two spatial frequencies
    (1/8 and 1/4 cycles/pixel). Each channel uses cosine/sine quadrature
    atoms; channel energy is response_cos^2 + response_sin^2. No tuning.
    """
    yy,xx=np.mgrid[0:m,0:m].astype(np.float64)
    xx=xx-(m-1)/2; yy=yy-(m-1)/2
    sigma=2.0; env=np.exp(-(xx*xx+yy*yy)/(2*sigma*sigma))
    cos_atoms=[]; sin_atoms=[]
    for theta in (0.0,np.pi/4,np.pi/2,3*np.pi/4):
        xr=xx*np.cos(theta)+yy*np.sin(theta)
        for freq in (1/8,1/4):
            c=env*np.cos(2*np.pi*freq*xr); s=env*np.sin(2*np.pi*freq*xr)
            c=c-c.mean(); s=s-s.mean()
            nc=np.linalg.norm(c); ns=np.linalg.norm(s)
            if nc<=0 or ns<=0: raise RuntimeError('Degenerate Gabor atom')
            cos_atoms.append(c/nc); sin_atoms.append(s/ns)
    return np.stack(cos_atoms),np.stack(sin_atoms)

_GABOR8_C,_GABOR8_S=_gabor_atoms(8)

def gabor_energy_descriptor(y: np.ndarray, m: int = 8):
    if m!=8: raise ValueError('Additional validation Gabor control is frozen to m=8')
    blocks=_blocks_centered(y,m)
    # inner products: blocks x atoms; square and average across local blocks
    rc=np.einsum('bij,kij->bk',blocks,_GABOR8_C,optimize=True)
    rs=np.einsum('bij,kij->bk',blocks,_GABOR8_S,optimize=True)
    return np.mean(rc*rc+rs*rs,axis=0).astype(np.float64)


def energies_to_ilr(E: np.ndarray, eta: float = 1e-6):
    P=np.vstack([smooth_composition(e,eta) for e in np.asarray(E,float)])
    V=helmert_basis(P.shape[1])
    Z=np.log(P)@V
    return P,Z


def extract_handcrafted_energy_chunked(manifest: pd.DataFrame, out_dir, chunk_size=128, m=8):
    out=Path(out_dir); chunks=out/'chunks'; chunks.mkdir(parents=True,exist_ok=True)
    n=len(manifest)
    for c0 in range(0,n,chunk_size):
        c1=min(n,c0+chunk_size); cp=chunks/f'chunk_{c0:06d}_{c1:06d}.npz'
        if cp.exists(): continue
        Ed=[]; Eg=[]; ids=[]
        for i in tqdm(range(c0,c1),desc=f'DWT/Gabor {c0}:{c1}',leave=False):
            _,y=load_preprocessed(manifest.filepath.iloc[i])
            Ed.append(dwt_energy_descriptor(y,m)); Eg.append(gabor_energy_descriptor(y,m)); ids.append(manifest.image_id.iloc[i])
        tmp=cp.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,image_id=np.array(ids),E_DWT=np.vstack(Ed),E_GABOR=np.vstack(Eg)); tmp.replace(cp)
    final=out/'handcrafted_energy.npz'
    if not final.exists():
        ids=[]; Ed=[]; Eg=[]
        for cp in sorted(chunks.glob('chunk_*.npz')):
            d=np.load(cp); ids.extend(d['image_id'].tolist()); Ed.append(d['E_DWT']); Eg.append(d['E_GABOR'])
        np.savez_compressed(final,image_id=np.array(ids),E_DWT=np.vstack(Ed),E_GABOR=np.vstack(Eg))
    return final


def clr_ilr_sanity(P: np.ndarray, max_pairs: int = 2000, seed: int = 2026):
    P=np.asarray(P,float); R=P.shape[1]
    L=np.log(P); clr=L-L.mean(axis=1,keepdims=True); V=helmert_basis(R); ilr=L@V
    rec=ilr@V.T
    rng=np.random.default_rng(seed); n=len(P); M=min(max_pairs,max(1,n*(n-1)//2))
    a=rng.integers(0,n,size=M); b=rng.integers(0,n,size=M)
    dc=np.linalg.norm(clr[a]-clr[b],axis=1); di=np.linalg.norm(ilr[a]-ilr[b],axis=1)
    cov_clr=np.cov(clr,rowvar=False); cov_ilr=np.cov(ilr,rowvar=False)
    return {
        'R':int(R), 'n':int(n),
        'max_abs_distance_diff':float(np.max(np.abs(dc-di))),
        'max_abs_clr_reconstruction_error':float(np.max(np.abs(clr-rec))),
        'clr_cov_rank':int(np.linalg.matrix_rank(cov_clr,tol=1e-10)),
        'ilr_cov_rank':int(np.linalg.matrix_rank(cov_ilr,tol=1e-10)),
    }

from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from scipy.fft import dctn
from tqdm.auto import tqdm
from ..preprocessing import load_preprocessed


def block_dct_coordinate_energy(y, m=8, offset=(0,0)):
    oy,ox=offset; h,w=y.shape
    hh=((h-oy)//m)*m; ww=((w-ox)//m)*m
    z=y[oy:oy+hh,ox:ox+ww]
    blocks=z.reshape(hh//m,m,ww//m,m).transpose(0,2,1,3).reshape(-1,m,m)
    blocks=blocks-blocks.mean(axis=(1,2),keepdims=True)
    C=dctn(blocks,type=2,norm="ortho",axes=(1,2))
    G=np.mean(C*C,axis=0)
    return G

def radial_band_map(m,R):
    u,v=np.meshgrid(np.arange(m),np.arange(m),indexing="ij")
    rho=np.sqrt(u*u+v*v)/(np.sqrt(2)*(m-1))
    b=np.ceil(rho*R).astype(int)-1
    b=np.clip(b,0,R-1); b[0,0]=-1
    return b

def band_counts(m,R):
    b=radial_band_map(m,R); return [int(np.sum(b==r)) for r in range(R)]

def energy_to_bands(G,R):
    b=radial_band_map(G.shape[0],R); return np.array([G[b==r].sum() for r in range(R)],dtype=np.float64)

def smooth_composition(E,eta=1e-6):
    E=np.asarray(E,dtype=np.float64); S=float(E.sum())
    if S<=0: return np.full(len(E),1/len(E),dtype=np.float64)
    ebar=E.mean(); z=E+eta*ebar; return z/z.sum()

def helmert_basis(R):
    # R x (R-1), orthonormal basis for 1^perp
    V=np.zeros((R,R-1),dtype=np.float64)
    for j in range(1,R):
        V[:j,j-1]=1/np.sqrt(j*(j+1)); V[j,j-1]=-j/np.sqrt(j*(j+1))
    return V

def ilr(p):
    p=np.asarray(p,dtype=np.float64); return helmert_basis(len(p)).T@np.log(p)

def spectral_from_G(Gs,R,eta=1e-6):
    E=np.vstack([energy_to_bands(G,R) for G in Gs]); P=np.vstack([smooth_composition(e,eta) for e in E]); V=helmert_basis(R); Z=np.log(P)@V
    return E,P,Z

def extract_coordinate_energy_chunked(manifest: pd.DataFrame,out_dir,chunk_size=256,m=8,offset=(0,0)):
    out=Path(out_dir); chunks=out/"chunks"; chunks.mkdir(parents=True,exist_ok=True); n=len(manifest)
    tag=f"o{offset[0]}_{offset[1]}"
    for c0 in range(0,n,chunk_size):
        c1=min(n,c0+chunk_size); cp=chunks/f"{tag}_chunk_{c0:06d}_{c1:06d}.npz"
        if cp.exists(): continue
        G=[]; ids=[]; SE=[]
        for i in tqdm(range(c0,c1),desc=f"DCT {c0}:{c1}",leave=False):
            _,y=load_preprocessed(manifest.filepath.iloc[i]); g=block_dct_coordinate_energy(y,m,offset); G.append(g); SE.append(float(g.sum())); ids.append(manifest.image_id.iloc[i])
        tmp=cp.with_suffix(".tmp.npz"); np.savez_compressed(tmp,image_id=np.array(ids),G=np.stack(G),S_E=np.array(SE)); tmp.replace(cp)
    final=out/f"coordinate_energy_{tag}.npz"
    if not final.exists():
        G=[]; ids=[]; se=[]
        for cp in sorted(chunks.glob(f"{tag}_chunk_*.npz")):
            d=np.load(cp); G.append(d["G"]); ids.extend(d["image_id"].tolist()); se.extend(d["S_E"].tolist())
        np.savez_compressed(final,image_id=np.array(ids),G=np.vstack(G),S_E=np.array(se))
    return final


def radial_sector_map(m, R, sectors=2):
    """Combined radial/angular partition for the directional sensitivity analysis.

    The directional control fixes sectors=2 with angular cells
    [0, pi/4) and [pi/4, pi/2].  The returned map is radial-major:
    bin = radial_index * sectors + sector_index.  DC is -1.
    """
    if sectors != 2:
        raise ValueError("Directional sensitivity analysis is frozen to exactly two angular sectors")
    u,v=np.meshgrid(np.arange(m),np.arange(m),indexing="ij")
    rho=np.sqrt(u*u+v*v)/(np.sqrt(2)*(m-1))
    rb=np.ceil(rho*R).astype(int)-1
    rb=np.clip(rb,0,R-1)
    theta=np.arctan2(v,u)
    sb=(theta>=np.pi/4).astype(int)
    out=rb*sectors+sb
    out[0,0]=-1
    return out


def radial_sector_counts(m, R, sectors=2):
    z=radial_sector_map(m,R,sectors)
    return [int(np.sum(z==j)) for j in range(R*sectors)]


def directional_energy_to_bands(G, R, sectors=2):
    z=radial_sector_map(G.shape[0],R,sectors)
    return np.array([G[z==j].sum() for j in range(R*sectors)],dtype=np.float64)


def directional_spectral_from_G(Gs, R, sectors=2, eta=1e-6):
    """Radial-sector compositional representation used in the directional sensitivity analysis."""
    E=np.vstack([directional_energy_to_bands(G,R,sectors) for G in Gs])
    P=np.vstack([smooth_composition(e,eta) for e in E])
    V=helmert_basis(R*sectors)
    Z=np.log(P)@V
    return E,P,Z

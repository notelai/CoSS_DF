from pathlib import Path
from cossdf.datasets.adapters import _kth_condition

def test_kth_condition_parser():
    s,p,i,im=_kth_condition("42a-scale_4_im_5_col.png")
    assert s==4 and im==5 and p=="left_22.5" and i=="top_45"
    s,p,i,im=_kth_condition("x-scale_10_im_12_col.png")
    assert s==10 and p=="right_22.5" and i=="ambient"

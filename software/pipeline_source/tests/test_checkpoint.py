from cossdf.checkpoints import CheckpointStore

def test_checkpoint_roundtrip(tmp_path):
    c=CheckpointStore(tmp_path,"x"); sig=c.signature(a=1)
    assert not c.is_done("s",sig)
    c.mark_done("s",sig,["a"])
    assert c.is_done("s",sig)
    assert not c.is_done("s","bad")

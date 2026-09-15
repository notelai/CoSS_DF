from cossdf.datasets import download as d

class Resp:
    def __init__(self, text, status=200): self.text=text; self.status_code=status
    def raise_for_status(self):
        if self.status_code >= 400: raise RuntimeError(self.status_code)


def test_kylberg_nested_zip_discovery(monkeypatch):
    root = "https://host/KylbergTextureDatasetV1/"
    child = root + "without-rotations-zip/"
    pages = {
        root: '<a href="without-rotations-zip/">without-rotations-zip</a>'
              '<a href="subsetOfKylbergTextureDataset-6classes-40samples.zip">subset.zip</a>',
        child: '<a href="KylbergTextureDataset-v1.0-without-rotations.zip">full.zip</a>',
    }
    def fake_get(url, **kwargs):
        return Resp(pages[url])
    monkeypatch.setattr(d.requests, "get", fake_get)
    got = d._discover_kylberg(root)
    assert got == child + "KylbergTextureDataset-v1.0-without-rotations.zip"

import numpy as np

from training.cpt_financial.pretrain import PageDataset


def _files(tmp_path, sizes, base):
    """Made-up token files: file k holds base + 1000*k, base + 1000*k + 1, ... so every token
    says which file it came from and where."""
    paths = []
    for k, n in enumerate(sizes):
        p = tmp_path / f"f{base}_{k}.bin"
        (np.arange(n, dtype=np.uint32) + base + 1000 * k).tofile(p)
        paths.append(p)
    return paths


def test_pages_have_the_right_length_and_match_the_file(tmp_path):
    ds = PageDataset(_files(tmp_path, [100, 300], base=0), page_len=16, num_pages=200, seed=0)
    assert len(ds) == 200
    for i in range(len(ds)):
        page = ds[i]
        ids = page["input_ids"]
        assert ids.dtype == np.int64 and len(ids) == 16 and (page["labels"] == ids).all()
        assert (np.diff(ids) == 1).all()                    # one contiguous stretch of one file
        assert ids[0] == 1000 * ds.which[i] + ds.start[i]   # exactly where the dataset says


def test_same_seed_same_pages_different_seed_different_pages(tmp_path):
    paths = _files(tmp_path, [100, 300], base=0)
    a, b, c = (PageDataset(paths, 16, 50, seed=s) for s in (1, 1, 2))
    assert [a[i]["input_ids"].tolist() for i in range(50)] == [b[i]["input_ids"].tolist() for i in range(50)]
    assert [a[i]["input_ids"].tolist() for i in range(50)] != [c[i]["input_ids"].tolist() for i in range(50)]


def test_pages_only_come_from_the_files_given(tmp_path):
    study = _files(tmp_path, [100, 300], base=0)          # tokens 0-1299
    _files(tmp_path, [100], base=500_000)                 # a "check pile" file, never passed in
    ds = PageDataset(study, page_len=16, num_pages=500, seed=0)
    assert max(ds[i]["input_ids"].max() for i in range(len(ds))) < 500_000
    edge = PageDataset(_files(tmp_path, [16], base=900_000), page_len=16, num_pages=3, seed=0)
    assert all(edge[i]["input_ids"].tolist() == list(range(900_000, 900_016)) for i in range(3))


def test_pages_read_counts_a_partial_last_update():
    from training.cpt_financial.pretrain import pages_read
    assert pages_read(382, 16, 6103) == 6103   # the real run: 381 full updates + one of 7 pages
    assert pages_read(20, 16, 6103) == 320     # the smoke run: 20 full updates

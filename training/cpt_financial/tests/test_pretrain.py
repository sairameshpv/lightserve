import numpy as np
import pytest

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


def _checks(rule, domain, general):
    """Feed made-up check scores to the stop rule one check at a time; the check it stopped at, or None."""
    from transformers import TrainerControl, TrainerState
    state, control = TrainerState(), TrainerControl()
    for n, (d, g) in enumerate(zip(domain, general), 1):
        state.log_history += [{"eval_domain_loss": d}, {"eval_general_loss": g}]
        rule.on_evaluate(None, state, control)
        if control.should_training_stop:
            return n
    return None


def test_stop_rule_stops_when_either_loss_rises_three_checks_in_a_row():
    pytest.importorskip("transformers")
    from training.cpt_financial.pretrain import make_stop_on_rise_callback
    rule = make_stop_on_rise_callback()
    assert _checks(rule, [1.5, 1.4, 1.3, 1.2, 1.1], [2.0, 1.9, 1.9, 1.8, 1.8]) is None   # improving
    assert _checks(rule, [1.5, 1.4, 1.3, 1.2], [1.9, 1.95, 2.0, 2.1]) == 4               # general (forgetting)
    assert _checks(rule, [1.3, 1.31, 1.32, 1.33], [1.9] * 4) == 4                        # domain
    assert _checks(rule, [1.3, 1.31, 1.32, 1.30, 1.31], [1.9] * 5) is None               # 2 rises, then a dip
    assert _checks(rule, [1.3] * 6, [1.9] * 6) is None                                   # flat isn't rising


def test_lora_config_scales_alpha_with_rank():
    pytest.importorskip("peft")
    from training.cpt_financial.pretrain import lora_config
    assert (lora_config().r, lora_config().lora_alpha) == (16, 32)   # the old run, unchanged
    big = lora_config(64)
    assert (big.r, big.lora_alpha) == (64, 128) and len(big.target_modules) == 7


def test_check_pages_come_only_from_the_held_out_files(tmp_path):
    from training.cpt_financial.evaluate import check_pages
    _files(tmp_path, [5000], base=0).pop().rename(tmp_path / "train_000.bin")        # study pile
    _files(tmp_path, [5000], base=500_000).pop().rename(tmp_path / "val_000.bin")    # check pile
    pages = check_pages(str(tmp_path), num_pages=5)
    assert len(pages) == 5 and all(pages[i]["input_ids"].min() >= 500_000 for i in range(5))
    again = check_pages(str(tmp_path), num_pages=5)                                  # fixed: seed 1
    assert all((pages[i]["input_ids"] == again[i]["input_ids"]).all() for i in range(5))

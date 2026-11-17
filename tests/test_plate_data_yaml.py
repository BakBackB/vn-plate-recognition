import yaml

from vnplate.config import PROJECT_ROOT


def _load():
    with open(PROJECT_ROOT / "configs" / "plate_data.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_train_and_val_declared():
    assert {"train", "val"} <= set(_load())


def test_no_missing_test_split_declared():
    # the downloaded dataset has no images/test; declaring it makes training fail
    d = _load()
    if "test" in d:
        assert (PROJECT_ROOT / d["path"] / d["test"]).is_dir()


def test_names_are_consecutive_ids_from_zero():
    names = _load()["names"]
    assert sorted(names) == list(range(len(names)))
    assert list(names.values()) == ["BSD", "BSV"]


def test_nc_matches_names_if_present():
    d = _load()
    if "nc" in d:
        assert d["nc"] == len(d["names"])
import json

from vnplate.config import _load_provinces


def _write(tmp_path, obj):
    p = tmp_path / "prov.json"
    p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return p


def test_dict_code_to_name_uses_keys(tmp_path):
    p = _write(tmp_path, {"11": "Cao Bằng", "59": "TP. Hồ Chí Minh"})
    assert _load_provinces(p) == frozenset({"11", "59"})


def test_plain_list_still_supported(tmp_path):
    assert _load_provinces(_write(tmp_path, ["30", 59])) == frozenset({"30", "59"})


def test_short_codes_are_zero_padded(tmp_path):
    assert _load_provinces(_write(tmp_path, {"8": "x"})) == frozenset({"08"})
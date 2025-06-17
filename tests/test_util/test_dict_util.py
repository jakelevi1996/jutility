from jutility import util

def test_format_dict():
    d = {"def": 123, "gh": "xy", "a": "bc"}
    assert util.format_dict(d) == "a='bc', def=123, gh='xy'"
    assert util.format_dict(d, ignore_keys="a") == "def=123, gh='xy'"
    assert util.format_dict(d, key_order=["gh", "def"]) == "gh='xy', def=123"
    assert util.format_dict(
        d,
        key_order=["gh", "def"],
        ignore_keys="gh",
    ) == "def=123"

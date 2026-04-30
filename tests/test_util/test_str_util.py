from jutility import util

def test_remove_whitespace():
    s = "  abc  def \n ghi  \n"
    assert util.remove_whitespace(s) == "abcdefghi"

def test_strings_equal_except_whitespace():
    s = "  abc  def \n ghi  \n"
    b1 = util.strings_equal_except_whitespace(s, "ab\ncd\nef\ngh\ni")
    b2 = util.strings_equal_except_whitespace(s, "ab\ncd\nef\ngh")

    assert isinstance(b1, bool)
    assert b1
    assert isinstance(b2, bool)
    assert not b2

def test_list_to_hex():
    assert util.list_to_hex([0.0, 0.4563, 1.0]) == "0074ff"
    assert util.list_to_hex([0.7250, 0.0, 1.0]) == "b800ff"

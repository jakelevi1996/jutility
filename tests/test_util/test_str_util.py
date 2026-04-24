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

from jutility import util

OUTPUT_DIR = util.get_test_output_dir("test_util/test_table")

def test_get_item():
    printer = util.Printer("test_get_item", OUTPUT_DIR)

    table = util.Table(
        util.Column("a", ".5f"),
        util.Column("b", "s"),
        util.Column("c", "i"),
        printer=printer,
    )
    table.update(a=3.14, b="abc", c=10)
    table.update(b="def")
    table.update(a=3.141592, b="gh", c=99)

    assert str(table) == (
        "A          | B          | C         \n"
        "---------- | ---------- | ----------\n"
        "   3.14000 |        abc |         10\n"
        "           |        def |           \n"
        "   3.14159 |         gh |         99"
    )

    assert table.get_data("a") == [3.14, 3.141592]
    assert table.get_data("b") == ["abc", "def", "gh"]
    assert table.get_data("c") == [10, 99]
    assert table.get_item("c", 0) == 10
    assert table.get_item("c", 1) == None
    assert table.get_item("c", 2) == 99
    assert table.get_item("c", -1) == 99

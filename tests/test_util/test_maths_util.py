from jutility import util

OUTPUT_DIR = util.get_test_output_dir("test_util/test_maths_util")

def test_get_pretty_steps():
    table = util.Table(
        util.Column("x",        width=-25),
        util.Column("steps_4",  width=-25),
        util.Column("steps_5",  width=-25),
        util.Column("steps_6",  width=-25),
        printer=util.Printer("test_get_pretty_steps", OUTPUT_DIR)
    )

    for x in range (100):
        table.update(
            x=x,
            steps_4=util.get_pretty_steps(x, 4),
            steps_5=util.get_pretty_steps(x, 5),
            steps_6=util.get_pretty_steps(x, 6),
        )

    for max_len in [4, 5, 6]:
        for i in range(max_len):
            assert util.get_pretty_steps(i, max_len) == list(range(i+1))

        for i in range(max_len, 100):
            s = util.get_pretty_steps(i, max_len)
            assert s[0] == 0
            assert s[-1] <= i
            assert len(s) <= max_len
            assert len(s) >= max_len // 2

    assert util.get_pretty_steps(37, 4) == [0, 10, 20, 30]
    assert util.get_pretty_steps(37, 5) == [0, 10, 20, 30]
    assert util.get_pretty_steps(37, 6) == [0, 10, 20, 30]

    assert util.get_pretty_steps(40, 4) == [0, 20, 40]
    assert util.get_pretty_steps(40, 5) == [0, 10, 20, 30, 40]
    assert util.get_pretty_steps(40, 6) == [0, 10, 20, 30, 40]

    assert util.get_pretty_steps(56, 4) == [0, 20, 40]
    assert util.get_pretty_steps(56, 5) == [0, 20, 40]
    assert util.get_pretty_steps(56, 6) == [0, 10, 20, 30, 40, 50]

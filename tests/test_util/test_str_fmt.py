from jutility import util

def test_floatformatter():
    x = 100/7

    f = util.FloatFormatter()
    s = f.format(x)
    assert s == "14.285714"
    assert len(s) == 9

    f = util.FloatFormatter(width=20)
    s = f.format(x)
    assert s == "           14.285714"
    assert len(s) == 20

    f = util.FloatFormatter(precision=10)
    s = f.format(x)
    assert s == "14.2857142857"
    assert len(s) == 13

    f = util.FloatFormatter(width=20, precision=10)
    s = f.format(x)
    assert s == "       14.2857142857"
    assert len(s) == 20

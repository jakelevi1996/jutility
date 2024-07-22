from jutility import plotting, util

class Experiment:
    def setup(self, n):
        return

    def run(self):
        raise NotImplementedError()

    def __repr__(self):
        return type(self).__name__

def time_sweep(
    *experiments: Experiment,
    n_list=None,
    num_repeats=5,
    printer=None,
    n_sigma=1,
    plot_name="Time complexity",
    dir_name=None,
):
    if n_list is None:
        n_list = util.log_range(10, 1000, 10)

    exp_dict = {
        repr(exp): exp
        for exp in experiments
    }
    data_dict = {
        repr(exp): util.NoisyData(log_space_data=True)
        for exp in experiments
    }
    timer = util.Timer(verbose_exit=False)
    table = util.Table(
        util.CountColumn("c", -5),
        util.TimeColumn("t"),
        util.Column("name", "s", max(len(s) for s in exp_dict.keys())),
        util.Column("n",    "s", len(str(int(max(n_list))))),
        util.Column("repeat"),
        util.Column("time_taken", "s", 11),
        printer=printer,
    )
    for exp_name in sorted(exp_dict.keys()):
        exp  = exp_dict[ exp_name]
        data = data_dict[exp_name]
        for n in sorted(set(int(n) for n in n_list)):
            for i in range(num_repeats):
                exp.setup(n)
                with timer:
                    exp.run()

                data.update(n, timer.time_taken)
                t = util.time_format(timer.time_taken, concise=True)
                table.update(name=exp_name, n=n, repeat=i, time_taken=t)

    cp = plotting.ColourPicker(len(experiments))
    plotting.plot(
        *[
            line
            for i, name in enumerate(sorted(exp_dict.keys()))
            for x1, y in [data_dict[name].get_all_data()]
            for x2, mean, ucb, lcb in [
                data_dict[name].get_statistics(n_sigma=n_sigma)
            ]
            for line in [
                plotting.Scatter(x1, y, label=name, a=0.5, z=20, color=cp(i)),
                plotting.Line(x2, mean,             a=1.0, z=30, c=cp(i)),
                plotting.FillBetween(x2, lcb, ucb,  a=0.2, z=10, c=cp(i)),
            ]
        ],
        xlabel="n",
        ylabel="Time (seconds)",
        log_xscale=True,
        log_yscale=True,
        legend=True,
        plot_name=plot_name,
        dir_name=dir_name,
    )

    return data_dict

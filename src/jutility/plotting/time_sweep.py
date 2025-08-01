from jutility import plotting, util

class Experiment:
    def setup(self, n):
        return

    def run(self):
        raise NotImplementedError()

    def __repr__(self):
        return util.format_type(type(self))

def run(
    *experiments: Experiment,
    n_list=None,
    num_repeats=5,
    printer=None,
    n_sigma=1,
    plot_name="Time complexity",
    dir_name=None,
):
    if n_list is None:
        n_list = util.log_range(10, 1000, 10, unique_integers=True)

    exp_dict = {
        repr(exp): exp
        for exp in experiments
    }
    data_dict = {
        repr(exp): plotting.NoisyData(log_y=True)
        for exp in experiments
    }
    timer = util.Timer(verbose=False)
    table = util.Table(
        util.CountColumn("c", -5),
        util.TimeColumn("t"),
        util.Column("name", "s", max(len(s) for s in exp_dict.keys())),
        util.Column("n",    "s", len(str(max(n_list)))),
        util.Column("repeat"),
        util.Column("time_taken", "s", 11),
        printer=printer,
    )
    for exp_name in sorted(exp_dict.keys()):
        exp  = exp_dict[ exp_name]
        data = data_dict[exp_name]
        for n in n_list:
            for i in range(num_repeats):
                exp.setup(n)
                with timer:
                    exp.run()

                data.update(n, timer.get_last())
                t = util.time_format(timer.get_last(), concise=True)
                table.update(name=exp_name, n=n, repeat=i, time_taken=t)

    cp = plotting.ColourPicker(len(experiments))
    plotting.plot(
        *[
            data_dict[name].plot(cp(i), name, n_sigma)
            for i, name in enumerate(sorted(exp_dict.keys()))
        ],
        xlabel="n",
        ylabel="Time (seconds)",
        log_x=True,
        log_y=True,
        legend=True,
        plot_name=plot_name,
        dir_name=dir_name,
    )

    return data_dict

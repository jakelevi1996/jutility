"""
MIT License

Copyright (c) 2022 JAKE LEVI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
from jutility import util, plotting

def get_range(val_lo, val_hi, val_num=10, log_space=False):
    if log_space:
        log_lo, log_hi = np.log([val_lo, val_hi])
        val_range = np.exp(np.linspace(log_lo, log_hi, val_num))
    else:
        val_range = np.linspace(val_lo, val_hi, val_num)

    return val_range

class Parameter:
    def __init__(
        self,
        name,
        default,
        val_range,
        val_format=None,
        log_x_axis=False,
        plot_axis_properties=None,
    ):
        self.name = name
        self.default = default
        self.val_range = val_range
        self.val_format = val_format

        if plot_axis_properties is None:
            plot_axis_properties = plotting.AxisProperties(
                xlabel=name,
                ylabel="Result",
                log_xscale=log_x_axis,
            )

        self.plot_axis_properties = plot_axis_properties

    def __repr__(self):
        return (
            "Parameter(name=%r, default=%r, range=%r)"
            % (self.name, self.default, self.val_range)
        )

class Experiment:
    def run(self, **kwargs):
        raise NotImplementedError()

class ParamSweeper:
    def __init__(
        self,
        experiment,
        *parameters,
        n_repeats=5,
        n_sigma=1,
        higher_is_better=True,
        print_every=1,
        print_every_level=0,
        printer=None,
    ):
        self._experiment = experiment
        self._param_list = parameters
        self._n_repeats = n_repeats
        self._n_sigma = n_sigma
        self._higher_is_better = higher_is_better

        self._params_to_results_dict = dict()
        if printer is None:
            printer = util.Printer()
        self._context = util.ExceptionContext(
            suppress_exceptions=True,
            printer=printer,
        )
        parameter_columns = [
            util.Column(parameter.name, parameter.val_format, width=10)
            for parameter in parameters
        ]
        self._table = util.Table(
            util.TimeColumn("t"),
            util.Column("repeat"),
            util.Column("result",           ".3f", width=10),
            util.Column("mean_result",      ".3f", width=10),
            util.Column("new_best_score",   ".3f", width=10),
            *parameter_columns,
            print_every=print_every,
            print_every_level=print_every_level,
            printer=printer,
        )

    def find_best_parameters(self):
        while True:
            self._has_updated_any_parameters = False

            for parameter in self._param_list:
                self.sweep_parameter(parameter, update_parameters=True)

            if not self._has_updated_any_parameters:
                break

        return self._get_default_param_dict()

    def sweep_parameter(self, parameter, update_parameters=True):
        param_dict = self._get_default_param_dict()
        val_results_dict = dict()

        for val in parameter.val_range:
            param_dict[parameter.name] = val
            param_tuple = tuple(sorted(param_dict.items()))

            if param_tuple not in self._params_to_results_dict:
                results_list = self._run_experiment(param_dict)
                self._params_to_results_dict[param_tuple] = results_list
            else:
                results_list = self._params_to_results_dict[param_tuple]

            val_results_dict[val] = results_list

        if update_parameters:
            best_param_val, score = self._get_best_param_val(val_results_dict)
            if parameter.default != best_param_val:
                parameter.default = best_param_val
                self._has_updated_any_parameters = True

                self._table.update(
                    new_best_score=score,
                    **self._get_default_param_dict(),
                    level=2,
                )

        return val_results_dict

    def tighten_ranges(self, new_num_vals=15):
        for param in self._param_list:
            if any(not util.is_numeric(v) for v in param.val_range):
                continue
            lo_candidates = [v for v in param.val_range if v < param.default]
            hi_candidates = [v for v in param.val_range if v > param.default]
            if len(lo_candidates) > 0:
                val_lo = max(lo_candidates)
            else:
                val_lo = param.default / 2
            if len(hi_candidates) > 0:
                val_hi = min(hi_candidates)
            else:
                val_hi = param.default * 2
            new_range = get_range(val_lo, val_hi, new_num_vals)
            param.val_range = np.sort(
                np.concatenate([new_range, [param.default]])
            )

    def plot(
        self,
        experiment_name="Experiment",
        dir_name=None,
        **plot_kwargs,
    ):
        filename_list = []
        for parameter in self._param_list:
            noisy_data = plotting.NoisyData()
            param_dict = self._get_default_param_dict()

            for val in parameter.val_range:
                param_dict[parameter.name] = val
                param_tuple = tuple(sorted(param_dict.items()))
                results_list = self._params_to_results_dict[param_tuple]
                for result in results_list:
                    noisy_data.update(val, result)

            param_dict[parameter.name] = parameter.default
            param_tuple = tuple(sorted(param_dict.items()))
            results_list = self._params_to_results_dict[param_tuple]
            mean_default = np.mean(results_list)
            std_default  = np.std( results_list)

            if self._higher_is_better:
                optimal_h = mean_default - (self._n_sigma * std_default)
            else:
                optimal_h = mean_default + (self._n_sigma * std_default)

            if util.is_numeric(parameter.default):
                param_default_str = "%.3g" % parameter.default
            else:
                param_default_str = str(parameter.default)

            plot_filename = plotting.plot(
                *noisy_data.get_lines(n_sigma=self._n_sigma),
                plotting.HVLine(
                    v=parameter.default,
                    h=optimal_h,
                    c="r",
                    ls="--",
                    label="Optimal value = %s" % param_default_str,
                    zorder=40,
                ),
                plot_name=(
                    "Parameter sweep results for %r, varying parameter %r"
                    % (experiment_name, parameter.name)
                ),
                dir_name=dir_name,
                legend=True,
                axis_properties=parameter.plot_axis_properties,
                **plot_kwargs,
            )
            filename_list.append(plot_filename)

        return filename_list

    def _get_default_param_dict(self):
        param_dict = {
            parameter.name: parameter.default
            for parameter in self._param_list
        }
        return param_dict

    def _run_experiment(self, experiment_param_dict):
        results_list = []
        for i in range(self._n_repeats):
            with self._context:
                score = self._experiment.run(**experiment_param_dict)
                results_list.append(score)

                self._table.update(
                    repeat=i,
                    result=score,
                    **experiment_param_dict,
                )

        if len(results_list) > 0:
            self._table.update(
                mean_result=np.mean(results_list),
                **experiment_param_dict,
                level=1,
            )

        return results_list

    def _get_best_param_val(self, val_results_dict):
        non_empty_results_dict = {
            val: results_list
            for val, results_list in val_results_dict.items()
            if len(results_list) > 0
        }
        if self._higher_is_better:
            score_dict = {
                val: (np.mean(results) - self._n_sigma * np.std(results))
                for val, results in non_empty_results_dict.items()
            }
            best_param_val = max(
                non_empty_results_dict.keys(),
                key=lambda val: score_dict[val],
            )
        else:
            score_dict = {
                val: (np.mean(results) + self._n_sigma * np.std(results))
                for val, results in non_empty_results_dict.items()
            }
            best_param_val = min(
                non_empty_results_dict.keys(),
                key=lambda val: score_dict[val],
            )

        return best_param_val, score_dict[best_param_val]

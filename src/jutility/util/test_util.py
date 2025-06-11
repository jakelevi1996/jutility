import os
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR  = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
TEST_DIR    = os.path.join(SOURCE_DIR, "tests")

class Seeder:
    def __init__(self):
        self._used_seeds = set()

    def get_seed(self, *args) -> int:
        seed = sum(i * ord(c) for i, c in enumerate(str(args), start=1))
        while seed in self._used_seeds:
            seed += 1

        self._used_seeds.add(seed)
        return seed

    def get_rng(self, *args) -> np.random.Generator:
        seed = self.get_seed(*args)
        rng = np.random.default_rng(seed)
        return rng

def get_numpy_rng(*args) -> np.random.Generator:
    return Seeder().get_rng(*args)

def get_test_output_dir(*subdir_names) -> str:
    return os.path.join(TEST_DIR, "outputs", *subdir_names)

def check_type(instance, expected_type, name=None):
    if not isinstance(instance, expected_type):
        name_str = ("`%s` = " % name) if (name is not None) else ""
        exp_type_name = expected_type.__name__
        inst_type_name = type(instance).__name__
        error_msg = (
            "Expected %sinstance of `%s`, but received %sinstance of `%s`"
            % (name_str, exp_type_name, name_str, inst_type_name)
        )
        raise TypeError(error_msg)

def check_equal(value, expected_value, name=None):
    if value != expected_value:
        name_str = ("%s == " % name) if (name is not None) else ""
        error_msg = (
            "Expected `%s%s`, but received `%s%s`"
            % (name_str, expected_value, name_str, value)
        )
        raise RuntimeError(error_msg)

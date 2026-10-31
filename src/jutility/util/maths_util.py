import math

def get_pretty_steps(x: int, max_len: int) -> list[int]:
    step_bases = [1, 2, 5]
    for scale in range(math.ceil(math.log10(max(x, 1)))):
        for sb in step_bases:
            step = sb * (10 ** scale)
            steps = list(range(0, x + 1, step))
            if len(steps) <= max_len:
                return steps

    return sorted(set([0, x]))

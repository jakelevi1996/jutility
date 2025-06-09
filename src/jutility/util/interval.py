from jutility.util.time_util import Timer

class _Interval:
    def ready(self):
        raise NotImplementedError()

    def reset(self):
        return

    def full_reset(self):
        return

class Always(_Interval):
    def ready(self):
        return True

class Never(_Interval):
    def ready(self):
        return False

class CountInterval(_Interval):
    def __init__(self, max_count):
        self._max_count = max_count
        self.full_reset()

    def ready(self):
        self._count += 1
        return self._count >= self._max_count

    def reset(self):
        self._count = 0

    def full_reset(self):
        self._count = self._max_count

class TimeInterval(_Interval):
    def __init__(self, num_seconds):
        self._num_seconds_interval = num_seconds
        self._num_seconds_limit = 0
        self._timer = Timer()

    def ready(self):
        return self._timer.get_time_taken() >= self._num_seconds_limit

    def reset(self):
        t = self._timer.get_time_taken()
        while self._num_seconds_limit < t:
            self._num_seconds_limit += self._num_seconds_interval

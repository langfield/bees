""" Test that ``Metrics.start_interval()`` works correctly. """
import time
import random

import numpy as np

from bees.timer import Timer

# pylint: disable=no-value-for-parameter, protected-access


def test_timer_computes_times_correctly() -> None:
    """ Tests whether ``Timer`` computes times correctly. """

    intervals = ["a", "b", "c", "d"]
    NUM_TRIALS = 10
    NUM_INTERVAL_EXECUTIONS = 100

    for _ in range(NUM_TRIALS):
        interval_executions = []
        interval_times = []
        expected_timed_intervals = {}

        # Generate test case and expected results.
        for _ in range(NUM_INTERVAL_EXECUTIONS):
            interval = random.choice(intervals)
            interval_time = random.random() * 0.01

            interval_executions.append(interval)
            interval_times.append(interval_time)

            if interval in expected_timed_intervals:
                expected_timed_intervals[interval].append(interval_time)
            else:
                expected_timed_intervals[interval] = [interval_time]

        # Run and time intervals.
        timer = Timer()
        for interval, interval_time in zip(interval_executions, interval_times):
            timer.start_interval(interval)
            time.sleep(interval_time)
            timer.end_interval(interval)

        # Check timings.
        for interval in intervals:
            current_expected_times = expected_timed_intervals[interval]
            current_actual_times = timer.timed_intervals[interval]
            assert len(current_expected_times) == len(current_actual_times)

            for i in range(len(current_expected_times)):
                assert abs(current_expected_times[i] - current_actual_times[i]) < 0.01

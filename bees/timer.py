#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Print live training debug output and do reward analysis. """
from typing import Dict, List
import time

import numpy as np

# pylint: disable=too-few-public-methods


class Timer:
    """ Timer object to measure various intervals. """

    def __init__(self) -> None:
        """ __init__ function for ``Timer`` class. """

        self.timed_intervals: Dict[str, List[float]] = {}
        self.current_interval = ""
        self.interval_start = 0.0

    def start_interval(self, interval: str) -> None:
        """
        Starts timing an interval.

        Parameters
        ----------
        interval : ``str``.
            Name of the interval to start timing. Must not be empty string.
        """

        current_time = time.time()

        # This is here because the value of ``self.current_interval`` is "" between
        # intervals.
        assert interval != ""

        if self.current_interval == "":

            # Set interval name and starting time.
            self.current_interval = interval
            self.interval_start = current_time

        else:
            raise ValueError(
                "Call to 'Timer.start_interval()' before previous interval ended. "
                "Make sure to call 'Timer.end_interval()' between each call to "
                "'Timer.start_interval()'."
            )

    def end_interval(self, interval: str) -> None:
        """
        Finishes timing an interval, computes and stores time.

        Parameters
        ----------
        interval : ``str``.
            Name of interval to finish timing. Must not be empty string.
        """

        current_time = time.time()

        # This is here because the value of ``self.current_interval`` is "" between
        # intervals.
        assert interval != ""

        if interval == self.current_interval:

            # Compute interval time.
            interval_time = current_time - self.interval_start
            if interval in self.timed_intervals:
                self.timed_intervals[interval].append(interval_time)
            else:
                self.timed_intervals[interval] = [interval_time]

            # Reset state.
            self.current_interval = ""
            self.interval_start = 0.0

        else:
            raise ValueError(
                "Interval name '%s' does not match current interval '%s'."
                % (interval, self.current_interval)
            )

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """ Returns a summary of the current timed interval values.  """

        summary: Dict[str, Dict[str, float]] = {}
        total_times: Dict[str, float] = {}
        avg_times: Dict[str, float] = {}

        for interval, times in self.timed_intervals.items():
            total_times[interval] = sum(times)
            avg_times[interval] = np.mean(times)

        total_time = sum(list(total_times.values()))
        for interval in self.timed_intervals:
            summary[interval] = {}
            summary[interval]["total"] = total_times[interval]
            summary[interval]["average"] = avg_times[interval]
            summary[interval]["percentage"] = total_times[interval] / total_time

        return summary

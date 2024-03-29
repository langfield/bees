#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Object to send data between processes. """
import torch.multiprocessing as mp


class Pipe:
    """ Multiprocessing pipes. """

    def __init__(self) -> None:
        self.env_spout, self.env_funnel = mp.Pipe()
        self.action_spout, self.action_funnel = mp.Pipe()
        self.action_dist_spout, self.action_dist_funnel = mp.Pipe()
        self.loss_spout, self.loss_funnel = mp.Pipe()
        self.save_spout, self.save_funnel = mp.Pipe()

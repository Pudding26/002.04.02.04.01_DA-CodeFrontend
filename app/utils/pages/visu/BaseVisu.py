from abc import ABC, abstractmethod
import pandas as pd


class BaseVisu:
    def __init__(self, df, shared_state: dict, initial_config: dict = None):
        self.df = df
        self.shared_state = shared_state
        self.initial_config = initial_config or {}

    def render(self):
        raise NotImplementedError("Each visualization must implement `render()`")

    def get_state(self):
        return {}

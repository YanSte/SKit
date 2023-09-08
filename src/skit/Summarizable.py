from abc import ABC, abstractmethod

class Summarizable(ABC):
    def __init__(self, debug_mode=False):
        self._debug_mode = debug_mode

    def summary(self):
        self._print_sep()
        print(f"\n{self.__class__.__name__} Configuration Summary\n")
        for attr, value in self.__dict__.items():
            print(f"{attr}: {value}")
        self._print_sep()

    def _debug(self, *args):
        if self._debug_mode:
            print(*args)

    def _debug_sep(self, *args):
        if self._debug_mode:
            self._print_sep()

    def _print_sep(self, character="=", length=40):
        separator = character * length
        print(separator)

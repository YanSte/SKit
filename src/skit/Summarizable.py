from abc import ABC, abstractmethod

class Summarizable(ABC):
    def summary(self):
        self.__print_sep()
        print(f"\n{self.__class__.__name__} Configuration Summary\n")
        for attr, value in self.__dict__.items():
            print(f"{attr}: {value}")
        self.__print_sep()

    def __print_sep(self, character="=", length=50):
        separator = character * length
        print(separator)

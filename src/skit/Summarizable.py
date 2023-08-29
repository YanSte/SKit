from abc import ABC, abstractmethod

class Summarizable(ABC):
    def summary(self):
        title = f"=== {self.__class__.__name__} Configuration Summary ==="
        print(title)
        for attr, value in self.__dict__.items():
            print(f"{attr}: {value}")
        print("=" * len(title))

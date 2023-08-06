from typing import Any


class A():
    def __init__(self) -> None:
        self.__a = 1
        self.b = 2 + self.__a
    
    def __getattribute__(self, n: str) -> Any:
        print(n)
        return super().__getattribute__(n)

a = A()
a.b
a.__a
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Node:
    x: int
    y: int
    time: int
    g: float = field(compare=False)
    h: float = field(compare=False)
    f: float = field(compare=False)
    remaining_battery: float = field(compare=False)
    parent: Optional['Node'] = field(default=None, compare=False, repr=False)

    def __lt__(self, other):
        if self.f != other.f: return self.f < other.f
        if self.h != other.h: return self.h < other.h
        return self.remaining_battery > other.remaining_battery
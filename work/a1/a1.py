"""A1：T1–T5 的验收入口。

实现按题分散在 T1.py…T5.py（学习者的写法），这里只做转发，
不含任何实现——目的是让 `just check a1` 能找到这几个名字。

T6 是纸面改错题，行间批注放在 T6-改错题.md，不在这里。
"""

from T1 import Counter
from T2 import Vec2
from T3 import batch_indices
from T4 import ScaledShift
from T5 import Timer

__all__ = ["Counter", "Vec2", "batch_indices", "ScaledShift", "Timer"]

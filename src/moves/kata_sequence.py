from dataclasses import dataclass, field
from typing import Callable, Optional, Set, Tuple

Event = Tuple[str, Optional[str]]  # ("upper_block", "left") etc.

@dataclass
class KataStep:
    name: str
    detector: Callable[..., Optional[Event]]
    required_sides: Set[str] = field(default_factory=set)   # {"left","right"} or empty
    confirm_frames: int = 3
    _seen_sides: Set[str] = field(default_factory=set, init=False)
    _streak: int = field(default=0, init=False)

    def reset_runtime(self):
        self._seen_sides.clear()
        self._streak = 0

    def update(self, event: Optional[Event]) -> bool:
        if event is None:
            self._streak = 0
            return False

        move, side = event

        # ✅ critical: ignore detections for other moves
        if move != self.name:
            self._streak = 0
            return False

        # this step got a valid detection this frame
        self._streak += 1

        # track sides
        if side == "both":
            self._seen_sides.update({"left", "right"})
        elif side:
            self._seen_sides.add(side)

        # debounce gate
        if self._streak < self.confirm_frames:
            return False

        # step completion logic
        if not self.required_sides:
            return True

        return self.required_sides.issubset(self._seen_sides)

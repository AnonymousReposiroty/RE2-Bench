import typing as t

class _NumberRangeBase(_NumberParamTypeBase):

    def __init__(self, min: float | None=None, max: float | None=None, min_open: bool=False, max_open: bool=False, clamp: bool=False) -> None:
        self.min = min
        self.max = max
        self.min_open = min_open
        self.max_open = max_open
        self.clamp = clamp

    def to_info_dict(self) -> dict[str, t.Any]:
        info_dict = super().to_info_dict()
        info_dict.update(min=self.min, max=self.max, min_open=self.min_open, max_open=self.max_open, clamp=self.clamp)
        return info_dict

import logging
import logging.handlers
from typing import IO, Any, NoReturn

class SphinxLoggerAdapter(logging.LoggerAdapter[logging.Logger]):
    """LoggerAdapter allowing ``type`` and ``subtype`` keywords."""
    KEYWORDS = ['type', 'subtype', 'location', 'nonl', 'color', 'once']

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        extra = kwargs.setdefault('extra', {})
        for keyword in self.KEYWORDS:
            if keyword in kwargs:
                extra[keyword] = kwargs.pop(keyword)
        return (msg, kwargs)

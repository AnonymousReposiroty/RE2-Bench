import logging
import logging.handlers
from typing import IO, Any, NoReturn

class SphinxLogRecord(logging.LogRecord):
    """Log record class supporting location"""
    prefix = ''
    location: Any = None

    def getMessage(self) -> str:
        message = super().getMessage()
        location = getattr(self, 'location', None)
        if location:
            message = f'{location}: {self.prefix}{message}'
        elif self.prefix not in message:
            message = self.prefix + message
        return message

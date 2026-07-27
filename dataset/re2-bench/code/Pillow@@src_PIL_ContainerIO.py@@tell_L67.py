from typing import IO, AnyStr, Generic, Literal

class ContainerIO(Generic[AnyStr]):
    """
    A file object that provides read access to a part of an existing
    file (for example a TAR file).
    """

    def __init__(self, file: IO[AnyStr], offset: int, length: int) -> None:
        """
        Create file object.

        :param file: Existing file.
        :param offset: Start of region, in bytes.
        :param length: Size of region, in bytes.
        """
        self.fh: IO[AnyStr] = file
        self.pos = 0
        self.offset = offset
        self.length = length
        self.fh.seek(offset)

    def tell(self) -> int:
        """
        Get current file pointer.

        :returns: Offset from start of region, in bytes.
        """
        return self.pos

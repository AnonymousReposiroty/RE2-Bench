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

    def read(self, n: int=0) -> AnyStr:
        """
        Read data.

        :param n: Number of bytes to read. If omitted or zero,
            read until end of region.
        :returns: An 8-bit string.
        """
        if n:
            n = min(n, self.length - self.pos)
        else:
            n = self.length - self.pos
        if not n:
            return b'' if 'b' in self.fh.mode else ''
        self.pos = self.pos + n
        return self.fh.read(n)

    def readline(self) -> AnyStr:
        """
        Read a line of text.

        :returns: An 8-bit string.
        """
        s: AnyStr = b'' if 'b' in self.fh.mode else ''
        newline_character = b'\n' if 'b' in self.fh.mode else '\n'
        while True:
            c = self.read(1)
            if not c:
                break
            s = s + c
            if c == newline_character:
                break
        return s

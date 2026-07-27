import os
import typing as t
from gettext import gettext as _

class Path(ParamType):
    """The ``Path`` type is similar to the :class:`File` type, but
    returns the filename instead of an open file. Various checks can be
    enabled to validate the type of file and permissions.

    :param exists: The file or directory needs to exist for the value to
        be valid. If this is not set to ``True``, and the file does not
        exist, then all further checks are silently skipped.
    :param file_okay: Allow a file as a value.
    :param dir_okay: Allow a directory as a value.
    :param readable: if true, a readable check is performed.
    :param writable: if true, a writable check is performed.
    :param executable: if true, an executable check is performed.
    :param resolve_path: Make the value absolute and resolve any
        symlinks. A ``~`` is not expanded, as this is supposed to be
        done by the shell only.
    :param allow_dash: Allow a single dash as a value, which indicates
        a standard stream (but does not open it). Use
        :func:`~click.open_file` to handle opening this value.
    :param path_type: Convert the incoming path value to this type. If
        ``None``, keep Python's default, which is ``str``. Useful to
        convert to :class:`pathlib.Path`.

    .. versionchanged:: 8.1
        Added the ``executable`` parameter.

    .. versionchanged:: 8.0
        Allow passing ``path_type=pathlib.Path``.

    .. versionchanged:: 6.0
        Added the ``allow_dash`` parameter.
    """
    envvar_list_splitter: t.ClassVar[str] = os.path.pathsep

    def __init__(self, exists: bool=False, file_okay: bool=True, dir_okay: bool=True, writable: bool=False, readable: bool=True, resolve_path: bool=False, allow_dash: bool=False, path_type: type[t.Any] | None=None, executable: bool=False):
        self.exists = exists
        self.file_okay = file_okay
        self.dir_okay = dir_okay
        self.readable = readable
        self.writable = writable
        self.executable = executable
        self.resolve_path = resolve_path
        self.allow_dash = allow_dash
        self.type = path_type
        if self.file_okay and (not self.dir_okay):
            self.name: str = _('file')
        elif self.dir_okay and (not self.file_okay):
            self.name = _('directory')
        else:
            self.name = _('path')

    def to_info_dict(self) -> dict[str, t.Any]:
        info_dict = super().to_info_dict()
        info_dict.update(exists=self.exists, file_okay=self.file_okay, dir_okay=self.dir_okay, writable=self.writable, readable=self.readable, allow_dash=self.allow_dash)
        return info_dict

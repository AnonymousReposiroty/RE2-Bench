from typing import Callable, Final, Iterable, Iterator, TypeVar, Union

class SpecifierSet(BaseSpecifier):
    """This class abstracts handling of a set of version specifiers.

    It can be passed a single specifier (``>=3.0``), a comma-separated list of
    specifiers (``>=3.0,!=3.1``), or no specifier at all.
    """

    def __init__(self, specifiers: str | Iterable[Specifier]='', prereleases: bool | None=None) -> None:
        """Initialize a SpecifierSet instance.

        :param specifiers:
            The string representation of a specifier or a comma-separated list of
            specifiers which will be parsed and normalized before use.
            May also be an iterable of ``Specifier`` instances, which will be used
            as is.
        :param prereleases:
            This tells the SpecifierSet if it should accept prerelease versions if
            applicable or not. The default of ``None`` will autodetect it from the
            given specifiers.

        :raises InvalidSpecifier:
            If the given ``specifiers`` are not parseable than this exception will be
            raised.
        """
        if isinstance(specifiers, str):
            split_specifiers = [s.strip() for s in specifiers.split(',') if s.strip()]
            self._specs = frozenset(map(Specifier, split_specifiers))
        else:
            self._specs = frozenset(specifiers)
        self._prereleases = prereleases

    @property
    def prereleases(self) -> bool | None:
        if self._prereleases is not None:
            return self._prereleases
        if not self._specs:
            return None
        if any((s.prereleases for s in self._specs)):
            return True
        return None

    @prereleases.setter
    def prereleases(self, value: bool | None) -> None:
        self._prereleases = value

    def __repr__(self) -> str:
        """A representation of the specifier set that shows all internal state.

        Note that the ordering of the individual specifiers within the set may not
        match the input string.

        >>> SpecifierSet('>=1.0.0,!=2.0.0')
        <SpecifierSet('!=2.0.0,>=1.0.0')>
        >>> SpecifierSet('>=1.0.0,!=2.0.0', prereleases=False)
        <SpecifierSet('!=2.0.0,>=1.0.0', prereleases=False)>
        >>> SpecifierSet('>=1.0.0,!=2.0.0', prereleases=True)
        <SpecifierSet('!=2.0.0,>=1.0.0', prereleases=True)>
        """
        pre = f', prereleases={self.prereleases!r}' if self._prereleases is not None else ''
        return f'<SpecifierSet({str(self)!r}{pre})>'

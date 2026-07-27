import time
from os import getenv
from collections.abc import Collection, Iterable, Iterator, Sequence, Set
from sphinx.application import Sphinx

def correct_copyright_year(_app: Sphinx, config: Config) -> None:
    """Correct values of copyright year that are not coherent with
    the SOURCE_DATE_EPOCH environment variable (if set)

    See https://reproducible-builds.org/specs/source-date-epoch/
    """
    if source_date_epoch := int(getenv('SOURCE_DATE_EPOCH', '0')):
        source_date_epoch_year = time.gmtime(source_date_epoch).tm_year
    else:
        return

    # If the current year is the replacement year, there's no work to do.
    # We also skip replacement years that are in the future.
    current_year = time.localtime().tm_year
    if current_year <= source_date_epoch_year:
        return

    current_yr = str(current_year)
    replace_yr = str(source_date_epoch_year)
    for k in ('copyright', 'epub_copyright'):
        if k in config:
            value: str | Sequence[str] = config[k]
            if isinstance(value, str):
                config[k] = _substitute_copyright_year(value, current_yr, replace_yr)
            else:
                items = (
                    _substitute_copyright_year(x, current_yr, replace_yr) for x in value
                )
                config[k] = type(value)(items)  # type: ignore[call-arg]

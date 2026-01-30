from sphinx.errors import ConfigError, ThemeError
from sphinx.locale import _, __
from sphinx.application import Sphinx
from sphinx.config import Config

def error_on_html_sidebars_string_values(app: Sphinx, config: Config) -> None:
    """Support removed in Sphinx 2."""
    errors = {}
    for pattern, pat_sidebars in config.html_sidebars.items():
        if isinstance(pat_sidebars, str):
            errors[pattern] = [pat_sidebars]
    if not errors:
        return
    msg = __(
        "Values in 'html_sidebars' must be a list of strings. "
        'At least one pattern has a string value: %s. '
        'Change to `html_sidebars = %r`.'
    )
    bad_patterns = ', '.join(map(repr, errors))
    fixed = config.html_sidebars | errors
    raise ConfigError(msg % (bad_patterns, fixed))

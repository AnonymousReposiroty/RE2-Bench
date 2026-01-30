import re
from sphinx.errors import ConfigError
from sphinx.locale import __
from sphinx.application import Sphinx
from sphinx.config import Config

def compile_linkcheck_allowed_redirects(app: Sphinx, config: Config) -> None:
    """Compile patterns to the regexp objects."""
    if config.linkcheck_allowed_redirects is _SENTINEL_LAR:
        return
    if not isinstance(config.linkcheck_allowed_redirects, dict):
        msg = __(
            f'Invalid value `{config.linkcheck_allowed_redirects!r}` in '
            'linkcheck_allowed_redirects. Expected a dictionary.'
        )
        raise ConfigError(msg)
    allowed_redirects = {}
    for url, pattern in config.linkcheck_allowed_redirects.items():
        try:
            allowed_redirects[re.compile(url)] = re.compile(pattern)
        except re.error as exc:
            logger.warning(
                __('Failed to compile regex in linkcheck_allowed_redirects: %r %s'),
                exc.pattern,
                exc.msg,
            )
    config.linkcheck_allowed_redirects = allowed_redirects

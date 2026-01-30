from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple
from sphinx.locale import _, __
from sphinx.util.tags import Tags

def _read_conf_py(conf_path: Path, *, overrides: dict[str, Any], tags: Tags) -> Config:
    """Create a Config object from a conf.py file."""
    namespace = eval_config_file(conf_path, tags)

    # Note: Old sphinx projects have been configured as "language = None" because
    #       sphinx-quickstart previously generated this by default.
    #       To keep compatibility, they should be fallback to 'en' for a while
    #       (This conversion should not be removed before 2025-01-01).
    if namespace.get('language', ...) is None:
        logger.warning(
            __(
                "Invalid configuration value found: 'language = None'. "
                'Update your configuration to a valid language code. '
                "Falling back to 'en' (English)."
            )
        )
        namespace['language'] = 'en'
    return Config(namespace, overrides)

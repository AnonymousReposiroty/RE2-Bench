import logging
import logging.handlers

def getLogger(name: str) -> SphinxLoggerAdapter:
    """Get logger wrapped by :class:`sphinx.util.logging.SphinxLoggerAdapter`.

    Sphinx logger always uses ``sphinx.*`` namespace to be independent from
    settings of root logger.  It ensures logging is consistent even if a
    third-party extension or imported application resets logger settings.

    Example usage::

        >>> from sphinx.util import logging
        >>> logger = logging.getLogger(__name__)
        >>> logger.info('Hello, this is an extension!')
        Hello, this is an extension!
    """
    # add sphinx prefix to name forcely
    logger = logging.getLogger(NAMESPACE + '.' + name)
    # Forcely enable logger
    logger.disabled = False
    # wrap logger by SphinxLoggerAdapter
    return SphinxLoggerAdapter(logger, {})

from sphinx.config import Config

def default_latex_docclass(config: Config) -> dict[str, str]:
    """Better default latex_docclass settings for specific languages."""
    if config.language == 'ja':
        if config.latex_engine == 'uplatex':
            return {'manual': 'ujbook', 'howto': 'ujreport'}
        else:
            return {'manual': 'jsbook', 'howto': 'jreport'}
    else:
        return {}

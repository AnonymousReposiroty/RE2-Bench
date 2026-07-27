from sphinx.util import logging, texescape
from sphinx.util.osutil import SEP, copyfile, make_filename_from_project
from sphinx.config import Config

def default_latex_documents(config: Config) -> list[tuple[str, str, str, str, str]]:
    """Better default latex_documents settings."""
    project = texescape.escape(config.project, config.latex_engine)
    author = texescape.escape(config.author, config.latex_engine)
    return [
        (
            config.root_doc,
            make_filename_from_project(config.project) + '.tex',
            texescape.escape_abbr(project),
            texescape.escape_abbr(author),
            config.latex_theme,
        )
    ]

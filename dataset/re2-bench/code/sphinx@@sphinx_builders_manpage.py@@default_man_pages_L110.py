from sphinx.util.osutil import ensuredir, make_filename_from_project
from sphinx.config import Config

def default_man_pages(config: Config) -> list[tuple[str, str, str, list[str], int]]:
    """Better default man_pages settings."""
    filename = make_filename_from_project(config.project)
    return [
        (
            config.root_doc,
            filename,
            f'{config.project} {config.release}',
            [config.author],
            1,
        )
    ]

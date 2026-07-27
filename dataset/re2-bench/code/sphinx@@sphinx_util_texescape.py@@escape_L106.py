def escape(s: str, latex_engine: str | None = None) -> str:
    """Escape text for LaTeX output."""
    if latex_engine in {'lualatex', 'xelatex'}:
        # unicode based LaTeX engine
        return s.translate(_tex_escape_map_without_unicode)
    else:
        return s.translate(_tex_escape_map)

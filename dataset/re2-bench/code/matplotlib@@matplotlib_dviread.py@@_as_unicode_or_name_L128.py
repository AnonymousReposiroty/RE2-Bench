from collections import namedtuple
import fontTools.agl
from matplotlib import _api, cbook, font_manager

class Text(namedtuple('Text', 'x y font glyph width')):
    """
    A glyph in the dvi file.

    In order to render the glyph, load the glyph at index ``text.index``
    from the font at ``text.font.resolve_path()`` with size ``text.font.size``,
    warped with ``text.font.effects``, then draw it at position
    ``(text.x, text.y)``.

    ``text.glyph`` is the glyph number actually stored in the dvi file (whose
    interpretation depends on the font).  ``text.width`` is the glyph width in
    dvi units.
    """

    @property
    def index(self):
        """
        The FreeType index of this glyph (that can be passed to FT_Load_Glyph).
        """
        return self.font._index_dvi_to_freetype(self.glyph)
    font_path = property(lambda self: self.font.resolve_path())
    font_size = property(lambda self: self.font.size)
    font_effects = property(lambda self: self.font.effects)

    def _as_unicode_or_name(self):
        if self.font.subfont:
            raise NotImplementedError('Indexing TTC fonts is not supported yet')
        path = self.font.resolve_path()
        if path.name.lower().endswith('pk'):
            return f'{chr(self.glyph)}?' if chr(self.glyph).isprintable() else f'pk{self.glyph:#02x}'
        face = font_manager.get_font(path)
        glyph_name = face.get_glyph_name(self.index)
        glyph_str = fontTools.agl.toUnicode(glyph_name)
        return glyph_str or glyph_name

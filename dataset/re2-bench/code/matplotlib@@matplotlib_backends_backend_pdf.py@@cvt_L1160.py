from io import BytesIO
import itertools
import math
import os
import warnings
import matplotlib as mpl
from matplotlib import _api, _text_helpers, _type1font, cbook, dviread
from matplotlib.font_manager import get_font, fontManager as _fontManager
from matplotlib.ft2font import FT2Font, FaceFlags, Kerning, LoadFlags, StyleFlags
from . import _backend_pdf_ps
from encodings import cp1252

class PdfFile:
    """PDF file object."""

    def __init__(self, filename, metadata=None):
        """
        Parameters
        ----------
        filename : str or path-like or file-like
            Output target; if a string, a file will be opened for writing.

        metadata : dict from strings to strings and dates
            Information dictionary object (see PDF reference section 10.2.1
            'Document Information Dictionary'), e.g.:
            ``{'Creator': 'My software', 'Author': 'Me', 'Title': 'Awesome'}``.

            The standard keys are 'Title', 'Author', 'Subject', 'Keywords',
            'Creator', 'Producer', 'CreationDate', 'ModDate', and
            'Trapped'. Values have been predefined for 'Creator', 'Producer'
            and 'CreationDate'. They can be removed by setting them to `None`.
        """
        super().__init__()
        self._object_seq = itertools.count(1)
        self.xrefTable = [[0, 65535, 'the zero object']]
        self.passed_in_file_object = False
        self.original_file_like = None
        self.tell_base = 0
        fh, opened = cbook.to_filehandle(filename, 'wb', return_opened=True)
        if not opened:
            try:
                self.tell_base = filename.tell()
            except OSError:
                fh = BytesIO()
                self.original_file_like = filename
            else:
                fh = filename
                self.passed_in_file_object = True
        self.fh = fh
        self.currentstream = None
        fh.write(b'%PDF-1.4\n')
        fh.write(b'%\xac\xdc \xab\xba\n')
        self.rootObject = self.reserveObject('root')
        self.pagesObject = self.reserveObject('pages')
        self.pageList = []
        self.fontObject = self.reserveObject('fonts')
        self._extGStateObject = self.reserveObject('extended graphics states')
        self.hatchObject = self.reserveObject('tiling patterns')
        self.gouraudObject = self.reserveObject('Gouraud triangles')
        self.XObjectObject = self.reserveObject('external objects')
        self.resourceObject = self.reserveObject('resources')
        root = {'Type': Name('Catalog'), 'Pages': self.pagesObject}
        self.writeObject(self.rootObject, root)
        self.infoDict = _create_pdf_info_dict('pdf', metadata or {})
        self._internal_font_seq = (Name(f'F{i}') for i in itertools.count(1))
        self._fontNames = {}
        self._dviFontInfo = {}
        self._character_tracker = _backend_pdf_ps.CharacterTracker()
        self.alphaStates = {}
        self._alpha_state_seq = (Name(f'A{i}') for i in itertools.count(1))
        self._soft_mask_states = {}
        self._soft_mask_seq = (Name(f'SM{i}') for i in itertools.count(1))
        self._soft_mask_groups = []
        self._hatch_patterns = {}
        self._hatch_pattern_seq = (Name(f'H{i}') for i in itertools.count(1))
        self.gouraudTriangles = []
        self._images = {}
        self._image_seq = (Name(f'I{i}') for i in itertools.count(1))
        self.markers = {}
        self.multi_byte_charprocs = {}
        self.paths = []
        self._annotations = []
        self.pageAnnotations = []
        procsets = [Name(x) for x in 'PDF Text ImageB ImageC ImageI'.split()]
        resources = {'Font': self.fontObject, 'XObject': self.XObjectObject, 'ExtGState': self._extGStateObject, 'Pattern': self.hatchObject, 'Shading': self.gouraudObject, 'ProcSet': procsets}
        self.writeObject(self.resourceObject, resources)
    fontNames = _api.deprecated('3.11')(property(lambda self: self._fontNames))
    type1Descriptors = _api.deprecated('3.11')(property(lambda _: {}))

    @staticmethod
    def _get_subsetted_psname(ps_name, charmap):
        return PdfFile._get_subset_prefix(frozenset(charmap.keys())) + ps_name

    def beginStream(self, id, len, extra=None, png=None):
        assert self.currentstream is None
        self.currentstream = Stream(id, len, self, extra, png)

    def endStream(self):
        if self.currentstream is not None:
            self.currentstream.end()
            self.currentstream = None

    def outputStream(self, ref, data, *, extra=None):
        self.beginStream(ref.id, None, extra)
        self.currentstream.write(data)
        self.endStream()

    def fontName(self, fontprop):
        """
        Select a font based on fontprop and return a name suitable for
        ``Op.selectfont``. If fontprop is a string, it will be interpreted
        as the filename of the font.
        """
        if isinstance(fontprop, str):
            filenames = [fontprop]
        elif mpl.rcParams['pdf.use14corefonts']:
            filenames = _fontManager._find_fonts_by_props(fontprop, fontext='afm', directory=RendererPdf._afm_font_dir)
        else:
            filenames = _fontManager._find_fonts_by_props(fontprop)
        first_Fx = None
        for fname in filenames:
            Fx = self._fontNames.get(fname)
            if not first_Fx:
                first_Fx = Fx
            if Fx is None:
                Fx = next(self._internal_font_seq)
                self._fontNames[fname] = Fx
                _log.debug('Assigning font %s = %r', Fx, fname)
                if not first_Fx:
                    first_Fx = Fx
        return first_Fx

    def _get_xobject_glyph_name(self, filename, glyph_name):
        Fx = self.fontName(filename)
        return '-'.join([Fx.name.decode(), os.path.splitext(os.path.basename(filename))[0], glyph_name])
    _identityToUnicodeCMap = b'/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n/CIDSystemInfo\n<< /Registry (Adobe)\n   /Ordering (UCS)\n   /Supplement 0\n>> def\n/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n1 begincodespacerange\n<0000> <ffff>\nendcodespacerange\n%d beginbfrange\n%s\nendbfrange\nendcmap\nCMapName currentdict /CMap defineresource pop\nend\nend'

    def embedTTF(self, filename, characters):
        """Embed the TTF font from the named file into the document."""
        font = get_font(filename)
        fonttype = mpl.rcParams['pdf.fonttype']

        def cvt(length, upe=font.units_per_EM, nearest=True):
            """Convert font coordinates to PDF glyph coordinates."""
            value = length / upe * 1000
            if nearest:
                return round(value)
            if value < 0:
                return math.floor(value)
            else:
                return math.ceil(value)

        def embedTTFType3(font, characters, descriptor):
            """The Type 3-specific part of embedding a Truetype font"""
            widthsObject = self.reserveObject('font widths')
            fontdescObject = self.reserveObject('font descriptor')
            fontdictObject = self.reserveObject('font dictionary')
            charprocsObject = self.reserveObject('character procs')
            differencesArray = []
            firstchar, lastchar = (0, 255)
            bbox = [cvt(x, nearest=False) for x in font.bbox]
            fontdict = {'Type': Name('Font'), 'BaseFont': ps_name, 'FirstChar': firstchar, 'LastChar': lastchar, 'FontDescriptor': fontdescObject, 'Subtype': Name('Type3'), 'Name': descriptor['FontName'], 'FontBBox': bbox, 'FontMatrix': [0.001, 0, 0, 0.001, 0, 0], 'CharProcs': charprocsObject, 'Encoding': {'Type': Name('Encoding'), 'Differences': differencesArray}, 'Widths': widthsObject}
            from encodings import cp1252

            def get_char_width(charcode):
                s = ord(cp1252.decoding_table[charcode])
                width = font.load_char(s, flags=LoadFlags.NO_SCALE | LoadFlags.NO_HINTING).horiAdvance
                return cvt(width)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                widths = [get_char_width(charcode) for charcode in range(firstchar, lastchar + 1)]
            descriptor['MaxWidth'] = max(widths)
            glyph_ids = []
            differences = []
            multi_byte_chars = set()
            for c in characters:
                ccode = c
                gind = font.get_char_index(ccode)
                glyph_ids.append(gind)
                glyph_name = font.get_glyph_name(gind)
                if ccode <= 255:
                    differences.append((ccode, glyph_name))
                else:
                    multi_byte_chars.add(glyph_name)
            differences.sort()
            last_c = -2
            for c, name in differences:
                if c != last_c + 1:
                    differencesArray.append(c)
                differencesArray.append(Name(name))
                last_c = c
            rawcharprocs = _get_pdf_charprocs(filename, glyph_ids)
            charprocs = {}
            for charname in sorted(rawcharprocs):
                stream = rawcharprocs[charname]
                charprocDict = {}
                if charname in multi_byte_chars:
                    charprocDict = {'Type': Name('XObject'), 'Subtype': Name('Form'), 'BBox': bbox}
                    stream = stream[stream.find(b'd1') + 2:]
                charprocObject = self.reserveObject('charProc')
                self.outputStream(charprocObject, stream, extra=charprocDict)
                if charname in multi_byte_chars:
                    name = self._get_xobject_glyph_name(filename, charname)
                    self.multi_byte_charprocs[name] = charprocObject
                else:
                    charprocs[charname] = charprocObject
            self.writeObject(fontdictObject, fontdict)
            self.writeObject(fontdescObject, descriptor)
            self.writeObject(widthsObject, widths)
            self.writeObject(charprocsObject, charprocs)
            return fontdictObject

        def embedTTFType42(font, characters, descriptor):
            """The Type 42-specific part of embedding a Truetype font"""
            fontdescObject = self.reserveObject('font descriptor')
            cidFontDictObject = self.reserveObject('CID font dictionary')
            type0FontDictObject = self.reserveObject('Type 0 font dictionary')
            cidToGidMapObject = self.reserveObject('CIDToGIDMap stream')
            fontfileObject = self.reserveObject('font file stream')
            wObject = self.reserveObject('Type 0 widths')
            toUnicodeMapObject = self.reserveObject('ToUnicode map')
            subset_str = ''.join((chr(c) for c in characters))
            _log.debug('SUBSET %s characters: %s', filename, subset_str)
            with _backend_pdf_ps.get_glyphs_subset(filename, subset_str) as subset:
                fontdata = _backend_pdf_ps.font_as_file(subset)
            _log.debug('SUBSET %s %d -> %d', filename, os.stat(filename).st_size, fontdata.getbuffer().nbytes)
            full_font = font
            font = FT2Font(fontdata)
            cidFontDict = {'Type': Name('Font'), 'Subtype': Name('CIDFontType2'), 'BaseFont': ps_name, 'CIDSystemInfo': {'Registry': 'Adobe', 'Ordering': 'Identity', 'Supplement': 0}, 'FontDescriptor': fontdescObject, 'W': wObject, 'CIDToGIDMap': cidToGidMapObject}
            type0FontDict = {'Type': Name('Font'), 'Subtype': Name('Type0'), 'BaseFont': ps_name, 'Encoding': Name('Identity-H'), 'DescendantFonts': [cidFontDictObject], 'ToUnicode': toUnicodeMapObject}
            descriptor['FontFile2'] = fontfileObject
            self.outputStream(fontfileObject, fontdata.getvalue(), extra={'Length1': fontdata.getbuffer().nbytes})
            cid_to_gid_map = ['\x00'] * 65536
            widths = []
            max_ccode = 0
            for c in characters:
                ccode = c
                gind = font.get_char_index(ccode)
                glyph = font.load_char(ccode, flags=LoadFlags.NO_SCALE | LoadFlags.NO_HINTING)
                widths.append((ccode, cvt(glyph.horiAdvance)))
                if ccode < 65536:
                    cid_to_gid_map[ccode] = chr(gind)
                max_ccode = max(ccode, max_ccode)
            widths.sort()
            cid_to_gid_map = cid_to_gid_map[:max_ccode + 1]
            last_ccode = -2
            w = []
            max_width = 0
            unicode_groups = []
            for ccode, width in widths:
                if ccode != last_ccode + 1:
                    w.append(ccode)
                    w.append([width])
                    unicode_groups.append([ccode, ccode])
                else:
                    w[-1].append(width)
                    unicode_groups[-1][1] = ccode
                max_width = max(max_width, width)
                last_ccode = ccode
            unicode_bfrange = []
            for start, end in unicode_groups:
                if start > 65535:
                    continue
                end = min(65535, end)
                unicode_bfrange.append(b'<%04x> <%04x> [%s]' % (start, end, b' '.join((b'<%04x>' % x for x in range(start, end + 1)))))
            unicode_cmap = self._identityToUnicodeCMap % (len(unicode_groups), b'\n'.join(unicode_bfrange))
            glyph_ids = []
            for ccode in characters:
                if not _font_supports_glyph(fonttype, ccode):
                    gind = full_font.get_char_index(ccode)
                    glyph_ids.append(gind)
            bbox = [cvt(x, nearest=False) for x in full_font.bbox]
            rawcharprocs = _get_pdf_charprocs(filename, glyph_ids)
            for charname in sorted(rawcharprocs):
                stream = rawcharprocs[charname]
                charprocDict = {'Type': Name('XObject'), 'Subtype': Name('Form'), 'BBox': bbox}
                stream = stream[stream.find(b'd1') + 2:]
                charprocObject = self.reserveObject('charProc')
                self.outputStream(charprocObject, stream, extra=charprocDict)
                name = self._get_xobject_glyph_name(filename, charname)
                self.multi_byte_charprocs[name] = charprocObject
            cid_to_gid_map = ''.join(cid_to_gid_map).encode('utf-16be')
            self.outputStream(cidToGidMapObject, cid_to_gid_map)
            self.outputStream(toUnicodeMapObject, unicode_cmap)
            descriptor['MaxWidth'] = max_width
            self.writeObject(cidFontDictObject, cidFontDict)
            self.writeObject(type0FontDictObject, type0FontDict)
            self.writeObject(fontdescObject, descriptor)
            self.writeObject(wObject, w)
            return type0FontDictObject
        ps_name = self._get_subsetted_psname(font.postscript_name, font.get_charmap())
        ps_name = ps_name.encode('ascii', 'replace')
        ps_name = Name(ps_name)
        pclt = font.get_sfnt_table('pclt') or {'capHeight': 0, 'xHeight': 0}
        post = font.get_sfnt_table('post') or {'italicAngle': (0, 0)}
        ff = font.face_flags
        sf = font.style_flags
        flags = 0
        symbolic = False
        if FaceFlags.FIXED_WIDTH in ff:
            flags |= 1 << 0
        if 0:
            flags |= 1 << 1
        if symbolic:
            flags |= 1 << 2
        else:
            flags |= 1 << 5
        if StyleFlags.ITALIC in sf:
            flags |= 1 << 6
        if 0:
            flags |= 1 << 16
        if 0:
            flags |= 1 << 17
        if 0:
            flags |= 1 << 18
        descriptor = {'Type': Name('FontDescriptor'), 'FontName': ps_name, 'Flags': flags, 'FontBBox': [cvt(x, nearest=False) for x in font.bbox], 'Ascent': cvt(font.ascender, nearest=False), 'Descent': cvt(font.descender, nearest=False), 'CapHeight': cvt(pclt['capHeight'], nearest=False), 'XHeight': cvt(pclt['xHeight']), 'ItalicAngle': post['italicAngle'][1], 'StemV': 0}
        if fonttype == 3:
            return embedTTFType3(font, characters, descriptor)
        elif fonttype == 42:
            return embedTTFType42(font, characters, descriptor)
    hatchPatterns = _api.deprecated('3.10')(property(lambda self: {k: (e, f, h) for k, (e, f, h, l) in self._hatch_patterns.items()}))

    def reserveObject(self, name=''):
        """
        Reserve an ID for an indirect object.

        The name is used for debugging in case we forget to print out
        the object with writeObject.
        """
        id = next(self._object_seq)
        self.xrefTable.append([None, 0, name])
        return Reference(id)

    def recordXref(self, id):
        self.xrefTable[id][0] = self.fh.tell() - self.tell_base

    def writeObject(self, object, contents):
        self.recordXref(object.id)
        object.write(contents, self)

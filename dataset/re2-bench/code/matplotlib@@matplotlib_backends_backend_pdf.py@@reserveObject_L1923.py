from io import BytesIO
import itertools
from matplotlib import _api, _text_helpers, _type1font, cbook, dviread
from . import _backend_pdf_ps

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
    _identityToUnicodeCMap = b'/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n/CIDSystemInfo\n<< /Registry (Adobe)\n   /Ordering (UCS)\n   /Supplement 0\n>> def\n/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n1 begincodespacerange\n<0000> <ffff>\nendcodespacerange\n%d beginbfrange\n%s\nendbfrange\nendcmap\nCMapName currentdict /CMap defineresource pop\nend\nend'
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

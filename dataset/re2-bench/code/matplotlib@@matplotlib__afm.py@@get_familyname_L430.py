import re

class AFM:

    def __init__(self, fh):
        """Parse the AFM file in file object *fh*."""
        self._header = _parse_header(fh)
        self._metrics, self._metrics_by_name = _parse_char_metrics(fh)
        self._kern, self._composite = _parse_optional(fh)

    def get_fullname(self):
        """Return the font full name, e.g., 'Times-Roman'."""
        name = self._header.get(b'FullName')
        if name is None:
            name = self._header[b'FontName']
        return name

    def get_familyname(self):
        """Return the font family name, e.g., 'Times'."""
        name = self._header.get(b'FamilyName')
        if name is not None:
            return name
        name = self.get_fullname()
        extras = '(?i)([ -](regular|plain|italic|oblique|bold|semibold|light|ultralight|extra|condensed))+$'
        return re.sub(extras, '', name)

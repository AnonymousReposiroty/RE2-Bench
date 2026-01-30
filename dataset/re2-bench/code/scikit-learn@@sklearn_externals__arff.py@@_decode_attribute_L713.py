class ArffDecoder:
    """An ARFF decoder."""

    def __init__(self):
        """Constructor."""
        self._conversors = []
        self._current_line = 0

    def _decode_attribute(self, s):
        """(INTERNAL) Decodes an attribute line.

        The attribute is the most complex declaration in an arff file. All
        attributes must follow the template::

             @attribute <attribute-name> <datatype>

        where ``attribute-name`` is a string, quoted if the name contains any
        whitespace, and ``datatype`` can be:

        - Numerical attributes as ``NUMERIC``, ``INTEGER`` or ``REAL``.
        - Strings as ``STRING``.
        - Dates (NOT IMPLEMENTED).
        - Nominal attributes with format:

            {<nominal-name1>, <nominal-name2>, <nominal-name3>, ...}

        The nominal names follow the rules for the attribute names, i.e., they
        must be quoted if the name contains whitespaces.

        This method must receive a normalized string, i.e., a string without
        padding, including the "\r
" characters.

        :param s: a normalized string.
        :return: a tuple (ATTRIBUTE_NAME, TYPE_OR_VALUES).
        """
        _, v = s.split(' ', 1)
        v = v.strip()
        m = _RE_ATTRIBUTE.match(v)
        if not m:
            raise BadAttributeFormat()
        name, type_ = m.groups()
        name = str(name.strip('"\''))
        if type_[:1] == '{' and type_[-1:] == '}':
            try:
                type_ = _parse_values(type_.strip('{} '))
            except Exception:
                raise BadAttributeType()
            if isinstance(type_, dict):
                raise BadAttributeType()
        else:
            type_ = str(type_).upper()
            if type_ not in ['NUMERIC', 'REAL', 'INTEGER', 'STRING']:
                raise BadAttributeType()
        return (name, type_)

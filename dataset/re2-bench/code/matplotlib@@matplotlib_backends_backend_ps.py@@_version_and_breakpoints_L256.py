import struct

def _version_and_breakpoints(loca, fontdata):
    """
    Read the version number of the font and determine sfnts breakpoints.

    When a TrueType font file is written as a Type 42 font, it has to be
    broken into substrings of at most 65535 bytes. These substrings must
    begin at font table boundaries or glyph boundaries in the glyf table.
    This function determines all possible breakpoints and it is the caller's
    responsibility to do the splitting.

    Helper function for _font_to_ps_type42.

    Parameters
    ----------
    loca : fontTools.ttLib._l_o_c_a.table__l_o_c_a or None
        The loca table of the font if available
    fontdata : bytes
        The raw data of the font

    Returns
    -------
    version : tuple[int, int]
        A 2-tuple of the major version number and minor version number.
    breakpoints : list[int]
        The breakpoints is a sorted list of offsets into fontdata; if loca is not
        available, just the table boundaries.
    """
    v1, v2, numTables = struct.unpack('>3h', fontdata[:6])
    version = (v1, v2)

    tables = {}
    for i in range(numTables):
        tag, _, offset, _ = struct.unpack('>4sIII', fontdata[12 + i*16:12 + (i+1)*16])
        tables[tag.decode('ascii')] = offset
    if loca is not None:
        glyf_breakpoints = {tables['glyf'] + offset for offset in loca.locations[:-1]}
    else:
        glyf_breakpoints = set()
    breakpoints = sorted({*tables.values(), *glyf_breakpoints, len(fontdata)})

    return version, breakpoints

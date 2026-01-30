import enum
import re
import types
import unicodedata
import string
from pyparsing import (
    Empty, Forward, Literal, Group, NotAny, OneOrMore, Optional,
    ParseBaseException, ParseException, ParseExpression, ParseFatalException,
    ParserElement, ParseResults, QuotedString, Regex, StringEnd, ZeroOrMore,
    pyparsing_common, nested_expr, one_of)
from ._mathtext_data import (
    latex_to_bakoma, stix_glyph_fixes, stix_virtual_fonts, tex2uni)
from collections.abc import Iterable

class Parser:
    """
    A pyparsing-based parser for strings containing math expressions.

    Raw text may also appear outside of pairs of ``$``.

    The grammar is based directly on that in TeX, though it cuts a few corners.
    """

    class _MathStyle(enum.Enum):
        DISPLAYSTYLE = 0
        TEXTSTYLE = 1
        SCRIPTSTYLE = 2
        SCRIPTSCRIPTSTYLE = 3
    _binary_operators = set('+ * - −\n      \\pm             \\sqcap                   \\rhd\n      \\mp             \\sqcup                   \\unlhd\n      \\times          \\vee                     \\unrhd\n      \\div            \\wedge                   \\oplus\n      \\ast            \\setminus                \\ominus\n      \\star           \\wr                      \\otimes\n      \\circ           \\diamond                 \\oslash\n      \\bullet         \\bigtriangleup           \\odot\n      \\cdot           \\bigtriangledown         \\bigcirc\n      \\cap            \\triangleleft            \\dagger\n      \\cup            \\triangleright           \\ddagger\n      \\uplus          \\lhd                     \\amalg\n      \\dotplus        \\dotminus                \\Cap\n      \\Cup            \\barwedge                \\boxdot\n      \\boxminus       \\boxplus                 \\boxtimes\n      \\curlyvee       \\curlywedge              \\divideontimes\n      \\doublebarwedge \\leftthreetimes          \\rightthreetimes\n      \\slash          \\veebar                  \\barvee\n      \\cupdot         \\intercal                \\amalg\n      \\circledcirc    \\circleddash             \\circledast\n      \\boxbar         \\obar                    \\merge\n      \\minuscolon     \\dotsminusdots\n      '.split())
    _relation_symbols = set('\n      = < > :\n      \\leq          \\geq          \\equiv       \\models\n      \\prec         \\succ         \\sim         \\perp\n      \\preceq       \\succeq       \\simeq       \\mid\n      \\ll           \\gg           \\asymp       \\parallel\n      \\subset       \\supset       \\approx      \\bowtie\n      \\subseteq     \\supseteq     \\cong        \\Join\n      \\sqsubset     \\sqsupset     \\neq         \\smile\n      \\sqsubseteq   \\sqsupseteq   \\doteq       \\frown\n      \\in           \\ni           \\propto      \\vdash\n      \\dashv        \\dots         \\doteqdot    \\leqq\n      \\geqq         \\lneqq        \\gneqq       \\lessgtr\n      \\leqslant     \\geqslant     \\eqgtr       \\eqless\n      \\eqslantless  \\eqslantgtr   \\lesseqgtr   \\backsim\n      \\backsimeq    \\lesssim      \\gtrsim      \\precsim\n      \\precnsim     \\gnsim        \\lnsim       \\succsim\n      \\succnsim     \\nsim         \\lesseqqgtr  \\gtreqqless\n      \\gtreqless    \\subseteqq    \\supseteqq   \\subsetneqq\n      \\supsetneqq   \\lessapprox   \\approxeq    \\gtrapprox\n      \\precapprox   \\succapprox   \\precnapprox \\succnapprox\n      \\npreccurlyeq \\nsucccurlyeq \\nsqsubseteq \\nsqsupseteq\n      \\sqsubsetneq  \\sqsupsetneq  \\nlesssim    \\ngtrsim\n      \\nlessgtr     \\ngtrless     \\lnapprox    \\gnapprox\n      \\napprox      \\approxeq     \\approxident \\lll\n      \\ggg          \\nparallel    \\Vdash       \\Vvdash\n      \\nVdash       \\nvdash       \\vDash       \\nvDash\n      \\nVDash       \\oequal       \\simneqq     \\triangle\n      \\triangleq         \\triangleeq         \\triangleleft\n      \\triangleright     \\ntriangleleft      \\ntriangleright\n      \\trianglelefteq    \\ntrianglelefteq    \\trianglerighteq\n      \\ntrianglerighteq  \\blacktriangleleft  \\blacktriangleright\n      \\equalparallel     \\measuredrightangle \\varlrtriangle\n      \\Doteq        \\Bumpeq       \\Subset      \\Supset\n      \\backepsilon  \\because      \\therefore   \\bot\n      \\top          \\bumpeq       \\circeq      \\coloneq\n      \\curlyeqprec  \\curlyeqsucc  \\eqcirc      \\eqcolon\n      \\eqsim        \\fallingdotseq \\gtrdot     \\gtrless\n      \\ltimes       \\rtimes       \\lessdot     \\ne\n      \\ncong        \\nequiv       \\ngeq        \\ngtr\n      \\nleq         \\nless        \\nmid        \\notin\n      \\nprec        \\nsubset      \\nsubseteq   \\nsucc\n      \\nsupset      \\nsupseteq    \\pitchfork   \\preccurlyeq\n      \\risingdotseq \\subsetneq    \\succcurlyeq \\supsetneq\n      \\varpropto    \\vartriangleleft \\scurel\n      \\vartriangleright \\rightangle \\equal     \\backcong\n      \\eqdef        \\wedgeq       \\questeq     \\between\n      \\veeeq        \\disin        \\varisins    \\isins\n      \\isindot      \\varisinobar  \\isinobar    \\isinvb\n      \\isinE        \\nisd         \\varnis      \\nis\n      \\varniobar    \\niobar       \\bagmember   \\ratio\n      \\Equiv        \\stareq       \\measeq      \\arceq\n      \\rightassert  \\rightModels  \\smallin     \\smallowns\n      \\notsmallowns \\nsimeq'.split())
    _arrow_symbols = set('\n     \\leftarrow \\longleftarrow \\uparrow \\Leftarrow \\Longleftarrow\n     \\Uparrow \\rightarrow \\longrightarrow \\downarrow \\Rightarrow\n     \\Longrightarrow \\Downarrow \\leftrightarrow \\updownarrow\n     \\longleftrightarrow \\updownarrow \\Leftrightarrow\n     \\Longleftrightarrow \\Updownarrow \\mapsto \\longmapsto \\nearrow\n     \\hookleftarrow \\hookrightarrow \\searrow \\leftharpoonup\n     \\rightharpoonup \\swarrow \\leftharpoondown \\rightharpoondown\n     \\nwarrow \\rightleftharpoons \\leadsto \\dashrightarrow\n     \\dashleftarrow \\leftleftarrows \\leftrightarrows \\Lleftarrow\n     \\Rrightarrow \\twoheadleftarrow \\leftarrowtail \\looparrowleft\n     \\leftrightharpoons \\curvearrowleft \\circlearrowleft \\Lsh\n     \\upuparrows \\upharpoonleft \\downharpoonleft \\multimap\n     \\leftrightsquigarrow \\rightrightarrows \\rightleftarrows\n     \\rightrightarrows \\rightleftarrows \\twoheadrightarrow\n     \\rightarrowtail \\looparrowright \\rightleftharpoons\n     \\curvearrowright \\circlearrowright \\Rsh \\downdownarrows\n     \\upharpoonright \\downharpoonright \\rightsquigarrow \\nleftarrow\n     \\nrightarrow \\nLeftarrow \\nRightarrow \\nleftrightarrow\n     \\nLeftrightarrow \\to \\Swarrow \\Searrow \\Nwarrow \\Nearrow\n     \\leftsquigarrow \\overleftarrow \\overleftrightarrow \\cwopencirclearrow\n     \\downzigzagarrow \\cupleftarrow \\rightzigzagarrow \\twoheaddownarrow\n     \\updownarrowbar \\twoheaduparrow \\rightarrowbar \\updownarrows\n     \\barleftarrow \\mapsfrom \\mapsdown \\mapsup \\Ldsh \\Rdsh\n     '.split())
    _spaced_symbols = _binary_operators | _relation_symbols | _arrow_symbols
    _punctuation_symbols = set(', ; . ! \\ldotp \\cdotp'.split())
    _overunder_symbols = set('\n       \\sum \\prod \\coprod \\bigcap \\bigcup \\bigsqcup \\bigvee\n       \\bigwedge \\bigodot \\bigotimes \\bigoplus \\biguplus\n       '.split())
    _overunder_functions = set('lim liminf limsup sup max min'.split())
    _dropsub_symbols = set('\\int \\oint \\iint \\oiint \\iiint \\oiiint \\iiiint'.split())
    _fontnames = set('rm cal it tt sf bf bfit default bb frak scr regular'.split())
    _function_names = set('\n      arccos csc ker min arcsin deg lg Pr arctan det lim sec arg dim\n      liminf sin cos exp limsup sinh cosh gcd ln sup cot hom log tan\n      coth inf max tanh'.split())
    _ambi_delims = set('\n      | \\| / \\backslash \\uparrow \\downarrow \\updownarrow \\Uparrow\n      \\Downarrow \\Updownarrow . \\vert \\Vert'.split())
    _left_delims = set('\n      ( [ \\{ < \\lfloor \\langle \\lceil \\lbrace \\leftbrace \\lbrack \\leftparen \\lgroup\n      '.split())
    _right_delims = set('\n      ) ] \\} > \\rfloor \\rangle \\rceil \\rbrace \\rightbrace \\rbrack \\rightparen \\rgroup\n      '.split())
    _delims = _left_delims | _right_delims | _ambi_delims
    _small_greek = set([unicodedata.name(chr(i)).split()[-1].lower() for i in range(ord('α'), ord('ω') + 1)])
    _latin_alphabets = set(string.ascii_letters)

    def __init__(self) -> None:
        p = types.SimpleNamespace()

        def set_names_and_parse_actions() -> None:
            for key, val in vars(p).items():
                if not key.startswith('_'):
                    if key not in ('token', 'placeable', 'auto_delim'):
                        val.set_name(key)
                    if hasattr(self, key):
                        val.set_parse_action(getattr(self, key))

        def csnames(group: str, names: Iterable[str]) -> Regex:
            ends_with_alpha = []
            ends_with_nonalpha = []
            for name in names:
                if name[-1].isalpha():
                    ends_with_alpha.append(name)
                else:
                    ends_with_nonalpha.append(name)
            return Regex('\\\\(?P<{group}>(?:{alpha})(?![A-Za-z]){additional}{nonalpha})'.format(group=group, alpha='|'.join(map(re.escape, ends_with_alpha)), additional='|' if ends_with_nonalpha else '', nonalpha='|'.join(map(re.escape, ends_with_nonalpha))))
        p.float_literal = Regex('[-+]?([0-9]+\\.?[0-9]*|\\.[0-9]+)')
        p.space = one_of(self._space_widths)('space')
        p.style_literal = one_of([str(e.value) for e in self._MathStyle])('style_literal')
        p.symbol = Regex("[a-zA-Z0-9 +\\-*/<>=:,.;!\\?&'@()\\[\\]|\\U00000080-\\U0001ffff]|\\\\[%${}\\[\\]_|]" + '|\\\\(?:{})(?![A-Za-z])'.format('|'.join(map(re.escape, tex2uni))))('sym').leave_whitespace()
        p.unknown_symbol = Regex('\\\\[A-Za-z]+')('name')
        p.font = csnames('font', self._fontnames)
        p.start_group = Optional('\\math' + one_of(self._fontnames)('font')) + '{'
        p.end_group = Literal('}')
        p.delim = one_of(self._delims)
        p.auto_delim = Forward()
        p.placeable = Forward()
        p.named_placeable = Forward()
        p.required_group = Forward()
        p.optional_group = Forward()
        p.token = Forward()
        p.named_placeable <<= p.placeable
        set_names_and_parse_actions()
        p.optional_group <<= '{' + ZeroOrMore(p.token)('group') + '}'
        p.required_group <<= '{' + OneOrMore(p.token)('group') + '}'
        p.customspace = cmd('\\hspace', '{' + p.float_literal('space') + '}')
        p.accent = csnames('accent', [*self._accent_map, *self._wide_accents]) - p.named_placeable('sym')
        p.function = csnames('name', self._function_names)
        p.group = p.start_group + ZeroOrMore(p.token)('group') + p.end_group
        p.unclosed_group = p.start_group + ZeroOrMore(p.token)('group') + StringEnd()
        p.frac = cmd('\\frac', p.required_group('num') + p.required_group('den'))
        p.dfrac = cmd('\\dfrac', p.required_group('num') + p.required_group('den'))
        p.binom = cmd('\\binom', p.required_group('num') + p.required_group('den'))
        p.genfrac = cmd('\\genfrac', '{' + Optional(p.delim)('ldelim') + '}' + '{' + Optional(p.delim)('rdelim') + '}' + '{' + p.float_literal('rulesize') + '}' + '{' + Optional(p.style_literal)('style') + '}' + p.required_group('num') + p.required_group('den'))
        p.sqrt = cmd('\\sqrt{value}', Optional('[' + OneOrMore(NotAny(']') + p.token)('root') + ']') + p.required_group('value'))
        p.overline = cmd('\\overline', p.required_group('body'))
        p.overset = cmd('\\overset', p.optional_group('annotation') + p.optional_group('body'))
        p.underset = cmd('\\underset', p.optional_group('annotation') + p.optional_group('body'))
        p.text = cmd('\\text', QuotedString('{', '\\', end_quote_char='}'))
        p.substack = cmd('\\substack', nested_expr(opener='{', closer='}', content=Group(OneOrMore(p.token)) + ZeroOrMore(Literal('\\\\').suppress()))('parts'))
        p.subsuper = Optional(p.placeable)('nucleus') + OneOrMore(one_of(['_', '^']) - p.placeable)('subsuper') + Regex("'*")('apostrophes') | Regex("'+")('apostrophes') | p.named_placeable('nucleus') + Regex("'*")('apostrophes')
        p.simple = p.space | p.customspace | p.font | p.subsuper
        p.token <<= p.simple | p.auto_delim | p.unclosed_group | p.unknown_symbol
        p.operatorname = cmd('\\operatorname', '{' + ZeroOrMore(p.simple)('name') + '}')
        p.boldsymbol = cmd('\\boldsymbol', '{' + ZeroOrMore(p.simple)('value') + '}')
        p.placeable <<= p.accent | p.symbol | p.function | p.operatorname | p.group | p.frac | p.dfrac | p.binom | p.genfrac | p.overset | p.underset | p.sqrt | p.overline | p.text | p.boldsymbol | p.substack
        mdelim = '\\middle' - (p.delim('mdelim') | Error('Expected a delimiter'))
        p.auto_delim <<= '\\left' - (p.delim('left') | Error('Expected a delimiter')) + ZeroOrMore(p.simple | p.auto_delim | mdelim)('mid') + '\\right' - (p.delim('right') | Error('Expected a delimiter'))
        p.math = OneOrMore(p.token)
        p.math_string = QuotedString('$', '\\', unquote_results=False)
        p.non_math = Regex('(?:(?:\\\\[$])|[^$])*').leave_whitespace()
        p.main = p.non_math + ZeroOrMore(p.math_string + p.non_math) + StringEnd()
        set_names_and_parse_actions()
        self._expression = p.main
        self._math_expression = p.math
        self._in_subscript_or_superscript = False
    float_literal = staticmethod(pyparsing_common.convert_to_float)
    _space_widths = {'\\,': 0.16667, '\\thinspace': 0.16667, '\\/': 0.16667, '\\>': 0.22222, '\\:': 0.22222, '\\;': 0.27778, '\\ ': 0.33333, '~': 0.33333, '\\enspace': 0.5, '\\quad': 1, '\\qquad': 2, '\\!': -0.16667}
    _accent_map = {'hat': '\\circumflexaccent', 'breve': '\\combiningbreve', 'bar': '\\combiningoverline', 'grave': '\\combininggraveaccent', 'acute': '\\combiningacuteaccent', 'tilde': '\\combiningtilde', 'dot': '\\combiningdotabove', 'ddot': '\\combiningdiaeresis', 'dddot': '\\combiningthreedotsabove', 'ddddot': '\\combiningfourdotsabove', 'vec': '\\combiningrightarrowabove', '"': '\\combiningdiaeresis', '`': '\\combininggraveaccent', "'": '\\combiningacuteaccent', '~': '\\combiningtilde', '.': '\\combiningdotabove', '^': '\\circumflexaccent', 'overrightarrow': '\\rightarrow', 'overleftarrow': '\\leftarrow', 'mathring': '\\circ'}
    _wide_accents = set('widehat widetilde widebar'.split())
    optional_group = required_group
    overset = underset = _genset

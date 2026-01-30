from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata

def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_node(toctree)

    app.add_node(desc)
    app.add_node(desc_signature)
    app.add_node(desc_signature_line)
    app.add_node(desc_content)
    app.add_node(desc_inline)

    app.add_node(desc_name)
    app.add_node(desc_addname)
    app.add_node(desc_type)
    app.add_node(desc_returns)
    app.add_node(desc_parameterlist)
    app.add_node(desc_type_parameter_list)
    app.add_node(desc_parameter)
    app.add_node(desc_type_parameter)
    app.add_node(desc_optional)
    app.add_node(desc_annotation)

    for n in SIG_ELEMENTS:
        app.add_node(n)

    app.add_node(versionmodified)
    app.add_node(seealso)
    app.add_node(productionlist)
    app.add_node(production)
    app.add_node(index)
    app.add_node(centered)
    app.add_node(acks)
    app.add_node(hlist)
    app.add_node(hlistcol)
    app.add_node(compact_paragraph)
    app.add_node(glossary)
    app.add_node(only)
    app.add_node(start_of_file)
    app.add_node(highlightlang)
    app.add_node(tabular_col_spec)
    app.add_node(pending_xref)
    app.add_node(number_reference)
    app.add_node(download_reference)
    app.add_node(literal_emphasis)
    app.add_node(literal_strong)
    app.add_node(manpage)

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }

import traceback
from importlib import import_module
from sphinx.domains import ObjType
from sphinx.errors import ExtensionError, SphinxError, VersionRequirementError
from sphinx.extension import Extension
from sphinx.locale import __
from sphinx.roles import XRefRole
from sphinx.util._pathlib import _StrPath
from sphinx.util.logging import prefixed_warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, TypeAlias
from docutils import nodes
from docutils.nodes import Element, Node, TextElement
from docutils.parsers import Parser
from docutils.parsers.rst import Directive
from docutils.transforms import Transform
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.domains import Domain, Index
from sphinx.ext.autodoc._legacy_class_based._documenters import Documenter
from sphinx.util.typing import (
    ExtensionMetadata,
    RoleFunction,
    TitleGetter,
    _ExtensionSetupFunc,
)

class SphinxComponentRegistry:

    def __init__(self) -> None:
        self.autodoc_attrgetters: dict[type, Callable[[Any, str, Any], Any]] = {}
        self.builders: dict[str, type[Builder]] = {}
        self.documenters: dict[str, type[Documenter]] = {}
        self.css_files: list[tuple[str, dict[str, Any]]] = []
        self.domains: dict[str, type[Domain]] = {}
        self.domain_directives: dict[str, dict[str, type[Directive]]] = {}
        self.domain_indices: dict[str, list[type[Index]]] = {}
        self.domain_object_types: dict[str, dict[str, ObjType]] = {}
        self.domain_roles: dict[str, dict[str, RoleFunction | XRefRole]] = {}
        self.enumerable_nodes: dict[type[Node], tuple[str, TitleGetter | None]] = {}
        self.html_inline_math_renderers: dict[str, _MathsInlineRenderers] = {}
        self.html_block_math_renderers: dict[str, _MathsBlockRenderers] = {}
        self.html_assets_policy: str = 'per_page'
        self.html_themes: dict[str, _StrPath] = {}
        self.js_files: list[tuple[str | None, dict[str, Any]]] = []
        self.latex_packages: list[tuple[str, str | None]] = []
        self.latex_packages_after_hyperref: list[tuple[str, str | None]] = []
        self.post_transforms: list[type[Transform]] = []
        self.source_parsers: dict[str, type[Parser]] = {}
        self.source_suffix: dict[str, str] = {}
        self.translators: dict[str, type[nodes.NodeVisitor]] = {}
        self.translation_handlers: dict[str, dict[str, _NodeHandlerPair]] = {}
        self.transforms: list[type[Transform]] = []

    def load_extension(self, app: Sphinx, extname: str) -> None:
        """Load a Sphinx extension."""
        if extname in app.extensions:
            return
        if extname in EXTENSION_BLACKLIST:
            logger.warning(__('the extension %r was already merged with Sphinx since version %s; this extension is ignored.'), extname, EXTENSION_BLACKLIST[extname])
            return
        prefix = __('while setting up extension %s:') % extname
        with prefixed_warnings(prefix):
            try:
                mod = import_module(extname)
            except ImportError as err:
                logger.verbose(__('Original exception:\n') + traceback.format_exc())
                raise ExtensionError(__('Could not import extension %s') % extname, err) from err
            setup: _ExtensionSetupFunc | None = getattr(mod, 'setup', None)
            if setup is None:
                logger.warning(__('extension %r has no setup() function; is it really a Sphinx extension module?'), extname)
                metadata: ExtensionMetadata = {}
            else:
                try:
                    metadata = setup(app)
                except VersionRequirementError as err:
                    raise VersionRequirementError(__('The %s extension used by this project needs at least Sphinx v%s; it therefore cannot be built with this version.') % (extname, err)) from err
            if metadata is None:
                metadata = {}
            elif not isinstance(metadata, dict):
                logger.warning(__('extension %r returned an unsupported object from its setup() function; it should return None or a metadata dictionary'), extname)
                metadata = {}
            app.extensions[extname] = Extension(extname, mod, **metadata)

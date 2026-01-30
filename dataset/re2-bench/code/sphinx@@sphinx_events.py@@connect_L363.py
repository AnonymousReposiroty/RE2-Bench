from collections import defaultdict
from typing import TYPE_CHECKING, NamedTuple, overload
from sphinx.errors import ExtensionError, SphinxError
from sphinx.locale import __
from collections.abc import Callable, Iterable, Sequence, Set
from pathlib import Path
from typing import Any, Literal
from docutils import nodes
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.config import Config
from sphinx.domains import Domain
from sphinx.environment import BuildEnvironment
from sphinx.ext.autodoc._event_listeners import (
    _AutodocBeforeProcessSignatureListener,
    _AutodocProcessBasesListener,
    _AutodocProcessDocstringListener,
    _AutodocProcessSignatureListener,
    _AutodocSkipMemberListener,
)
from sphinx.ext.todo import todo_node

class EventManager:
    """Event manager for Sphinx."""

    def __init__(self, app: Sphinx) -> None:
        self._app = app
        self.events = core_events.copy()
        self.listeners: dict[str, list[EventListener]] = defaultdict(list)
        self.next_listener_id = 0
        self._reraise_errors: bool = app.pdb

    @overload
    def connect(self, name: Literal['config-inited'], callback: Callable[[Sphinx, Config], None], priority: int) -> int:
        ...

    @overload
    def connect(self, name: Literal['builder-inited'], callback: Callable[[Sphinx], None], priority: int) -> int:
        ...

    @overload
    def connect(self, name: Literal['env-get-outdated'], callback: Callable[[Sphinx, BuildEnvironment, Set[str], Set[str], Set[str]], Sequence[str]], priority: int) -> int:
        ...

    @overload
    def connect(self, name: Literal['env-before-read-docs'], callback: Callable[[Sphinx, BuildEnvironment, list[str]], None], priority: int) -> int:
        ...

    @overload
    def connect(self, name: Literal['env-purge-doc'], callback: Callable[[Sphinx, BuildEnvironment, str], None], priority: int) -> int:
        ...

    @overload
    def connect(self, name: Literal['source-read'], callback: Callable[[Sphinx, str, list[str]], None], priority: int) -> int:
        ...

    @overload
    def connect(self, name: Literal['include-read'], callback: Callable[[Sphinx, Path, str, list[str]], None], priority: int) -> int:
        ...

    @overload
    def connect(self, name: Literal['doctree-read'], callback: Callable[[Sphinx, nodes.document], None], priority: int) -> int:
        ...

    @overload
    def connect(self, name: Literal['env-merge-info'], callback: Callable[[Sphinx, BuildEnvironment, Set[str], BuildEnvironment], None], priority: int) -> int:
        ...

    @overload
    def connect(self, name: Literal['env-updated'], callback: Callable[[Sphinx, BuildEnvironment], str], priority: int) -> int:
        ...

    @overload
    def connect(self, name: Literal['env-get-updated'], callback: Callable[[Sphinx, BuildEnvironment], Iterable[str]], priority: int) -> int:
        ...

    @overload
    def connect(self, name: Literal['env-check-consistency'], callback: Callable[[Sphinx, BuildEnvironment], None], priority: int) -> int:
        ...

    @overload
    def connect(self, name: Literal['write-started'], callback: Callable[[Sphinx, Builder], None], priority: int) -> int:
        ...

    @overload
    def connect(self, name: Literal['doctree-resolved'], callback: Callable[[Sphinx, nodes.document, str], None], priority: int) -> int:
        ...

    @overload
    def connect(self, name: Literal['missing-reference'], callback: Callable[[Sphinx, BuildEnvironment, addnodes.pending_xref, nodes.TextElement], nodes.reference | None], priority: int) -> int:
        ...

    @overload
    def connect(self, name: Literal['warn-missing-reference'], callback: Callable[[Sphinx, Domain, addnodes.pending_xref], bool | None], priority: int) -> int:
        ...

    @overload
    def connect(self, name: Literal['build-finished'], callback: Callable[[Sphinx, Exception | None], None], priority: int) -> int:
        ...

    @overload
    def connect(self, name: Literal['html-collect-pages'], callback: Callable[[Sphinx], Iterable[tuple[str, dict[str, Any], str]]], priority: int) -> int:
        ...

    @overload
    def connect(self, name: Literal['html-page-context'], callback: Callable[[Sphinx, str, str, dict[str, Any], nodes.document], str | None], priority: int) -> int:
        ...

    @overload
    def connect(self, name: Literal['linkcheck-process-uri'], callback: Callable[[Sphinx, str], str | None], priority: int) -> int:
        ...

    @overload
    def connect(self, name: Literal['object-description-transform'], callback: Callable[[Sphinx, str, str, addnodes.desc_content], None], priority: int) -> int:
        ...

    @overload
    def connect(self, name: Literal['autodoc-process-docstring'], callback: _AutodocProcessDocstringListener, priority: int) -> int:
        ...

    @overload
    def connect(self, name: Literal['autodoc-before-process-signature'], callback: _AutodocBeforeProcessSignatureListener, priority: int) -> int:
        ...

    @overload
    def connect(self, name: Literal['autodoc-process-signature'], callback: _AutodocProcessSignatureListener, priority: int) -> int:
        ...

    @overload
    def connect(self, name: Literal['autodoc-process-bases'], callback: _AutodocProcessBasesListener, priority: int) -> int:
        ...

    @overload
    def connect(self, name: Literal['autodoc-skip-member'], callback: _AutodocSkipMemberListener, priority: int) -> int:
        ...

    @overload
    def connect(self, name: Literal['todo-defined'], callback: Callable[[Sphinx, todo_node], None], priority: int) -> int:
        ...

    @overload
    def connect(self, name: Literal['viewcode-find-source'], callback: Callable[[Sphinx, str], tuple[str, dict[str, tuple[Literal['class', 'def', 'other'], int, int]]]], priority: int) -> int:
        ...

    @overload
    def connect(self, name: Literal['viewcode-follow-imported'], callback: Callable[[Sphinx, str, str], str | None], priority: int) -> int:
        ...

    @overload
    def connect(self, name: str, callback: Callable[..., Any], priority: int) -> int:
        ...

    def connect(self, name: str, callback: Callable[..., Any], priority: int) -> int:
        """Connect a handler to specific event.

        Register *callback* to be called when the *name* event is emitted.

        See :ref:`event callbacks <events>` for details on available core events
        and the arguments of their corresponding callback functions.

        :param name:
            The name of the target event.
        :param callback:
            Callback function for the event.
        :param priority:
            The priority of the callback.
            The callbacks will be invoked in ascending order of *priority*.
        :return:
            A listener ID, for use with the :meth:`disconnect` method.

        .. versionchanged:: 3.0

           Support *priority*
        """
        if name not in self.events:
            msg = __('Unknown event name: %s')
            raise ExtensionError(msg % name)
        listener_id = self.next_listener_id
        self.next_listener_id += 1
        self.listeners[name].append(EventListener(listener_id, callback, priority))
        return listener_id

import sphinx
from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata, OptionSpec

def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_event('todo-defined')
    app.add_config_value('todo_include_todos', False, 'html', types=frozenset({bool}))
    app.add_config_value('todo_link_only', False, 'html', types=frozenset({bool}))
    app.add_config_value('todo_emit_warnings', False, 'html', types=frozenset({bool}))

    app.add_node(todolist)
    app.add_node(
        todo_node,
        html=(visit_todo_node, depart_todo_node),
        latex=(latex_visit_todo_node, latex_depart_todo_node),
        text=(visit_todo_node, depart_todo_node),
        man=(visit_todo_node, depart_todo_node),
        texinfo=(visit_todo_node, depart_todo_node),
    )

    app.add_directive('todo', Todo)
    app.add_directive('todolist', TodoList)
    app.add_domain(TodoDomain)
    app.connect('doctree-resolved', TodoListProcessor)
    return {
        'version': sphinx.__display_version__,
        'env_version': 2,
        'parallel_read_safe': True,
    }

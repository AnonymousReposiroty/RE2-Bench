# Problem: matplotlib@@matplotlib_backends_registry.py@@resolve_gui_or_backend_L373
# Module: matplotlib.backends.registry
# Function: resolve_gui_or_backend
# Line: 373

from matplotlib.backends.registry import BackendRegistry


def test_input(pred_input):
    obj_ins = BackendRegistry()
    obj_ins._loaded_entry_points = False
    obj_ins._backend_to_gui_framework = {}
    obj_ins._name_to_module = {'notebook': 'nbagg'}
    obj_ins_pred = BackendRegistry()
    obj_ins_pred._loaded_entry_points = pred_input['self']['_loaded_entry_points']
    obj_ins_pred._backend_to_gui_framework = pred_input['self']['_backend_to_gui_framework']
    obj_ins_pred._name_to_module = pred_input['self']['_name_to_module']
    assert obj_ins.resolve_gui_or_backend(gui_or_backend = 'notebook')==obj_ins_pred.resolve_gui_or_backend(gui_or_backend = pred_input['args']['gui_or_backend']), 'Prediction failed!'

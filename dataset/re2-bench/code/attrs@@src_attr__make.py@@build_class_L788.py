import abc
import contextlib
import inspect
import itertools
import sys
import weakref
from collections.abc import Callable, Mapping
from functools import cached_property
from typing import Any, NamedTuple, TypeVar
from . import _compat, _config, setters
from ._compat import (
    PY_3_10_PLUS,
    PY_3_11_PLUS,
    PY_3_13_PLUS,
    _AnnotationExtractor,
    _get_annotations,
    get_generic_base,
)

class _ClassBuilder:
    """
    Iteratively build *one* class.
    """
    __slots__ = ('_add_method_dunders', '_attr_names', '_attrs', '_base_attr_map', '_base_names', '_cache_hash', '_cls', '_cls_dict', '_delete_attribs', '_frozen', '_has_custom_setattr', '_has_post_init', '_has_pre_init', '_is_exc', '_on_setattr', '_pre_init_has_args', '_repr_added', '_script_snippets', '_slots', '_weakref_slot', '_wrote_own_setattr')

    def __init__(self, cls: type, these, auto_attribs: bool, props: ClassProps, has_custom_setattr: bool):
        attrs, base_attrs, base_map = _transform_attrs(cls, these, auto_attribs, props.kw_only, props.collected_fields_by_mro, props.field_transformer)
        self._cls = cls
        self._cls_dict = dict(cls.__dict__) if props.is_slotted else {}
        self._attrs = attrs
        self._base_names = {a.name for a in base_attrs}
        self._base_attr_map = base_map
        self._attr_names = tuple((a.name for a in attrs))
        self._slots = props.is_slotted
        self._frozen = props.is_frozen
        self._weakref_slot = props.has_weakref_slot
        self._cache_hash = props.hashability is ClassProps.Hashability.HASHABLE_CACHED
        self._has_pre_init = bool(getattr(cls, '__attrs_pre_init__', False))
        self._pre_init_has_args = False
        if self._has_pre_init:
            pre_init_func = cls.__attrs_pre_init__
            pre_init_signature = inspect.signature(pre_init_func)
            self._pre_init_has_args = len(pre_init_signature.parameters) > 1
        self._has_post_init = bool(getattr(cls, '__attrs_post_init__', False))
        self._delete_attribs = not bool(these)
        self._is_exc = props.is_exception
        self._on_setattr = props.on_setattr_hook
        self._has_custom_setattr = has_custom_setattr
        self._wrote_own_setattr = False
        self._cls_dict['__attrs_attrs__'] = self._attrs
        self._cls_dict['__attrs_props__'] = props
        if props.is_frozen:
            self._cls_dict['__setattr__'] = _frozen_setattrs
            self._cls_dict['__delattr__'] = _frozen_delattrs
            self._wrote_own_setattr = True
        elif self._on_setattr in (_DEFAULT_ON_SETATTR, setters.validate, setters.convert):
            has_validator = has_converter = False
            for a in attrs:
                if a.validator is not None:
                    has_validator = True
                if a.converter is not None:
                    has_converter = True
                if has_validator and has_converter:
                    break
            if self._on_setattr == _DEFAULT_ON_SETATTR and (not (has_validator or has_converter)) or (self._on_setattr == setters.validate and (not has_validator)) or (self._on_setattr == setters.convert and (not has_converter)):
                self._on_setattr = None
        if props.added_pickling:
            self._cls_dict['__getstate__'], self._cls_dict['__setstate__'] = self._make_getstate_setstate()
        self._script_snippets: list[tuple[str, dict, Callable[[dict, dict], Any]]] = []
        self._repr_added = False
        if not hasattr(self._cls, '__module__') or not hasattr(self._cls, '__qualname__'):
            self._add_method_dunders = self._add_method_dunders_safe
        else:
            self._add_method_dunders = self._add_method_dunders_unsafe

    def _eval_snippets(self) -> None:
        """
        Evaluate any registered snippets in one go.
        """
        script = '\n'.join([snippet[0] for snippet in self._script_snippets])
        globs = {}
        for _, snippet_globs, _ in self._script_snippets:
            globs.update(snippet_globs)
        locs = _linecache_and_compile(script, _generate_unique_filename(self._cls, 'methods'), globs)
        for _, _, hook in self._script_snippets:
            hook(self._cls_dict, locs)

    def build_class(self):
        """
        Finalize class based on the accumulated configuration.

        Builder cannot be used after calling this method.
        """
        self._eval_snippets()
        if self._slots is True:
            cls = self._create_slots_class()
            self._cls.__attrs_base_of_slotted__ = weakref.ref(cls)
        else:
            cls = self._patch_original_class()
            if PY_3_10_PLUS:
                cls = abc.update_abstractmethods(cls)
        if getattr(cls, '__attrs_init_subclass__', None) and '__attrs_init_subclass__' not in cls.__dict__:
            cls.__attrs_init_subclass__()
        return cls

    def _patch_original_class(self):
        """
        Apply accumulated methods and return the class.
        """
        cls = self._cls
        base_names = self._base_names
        if self._delete_attribs:
            for name in self._attr_names:
                if name not in base_names and getattr(cls, name, _SENTINEL) is not _SENTINEL:
                    with contextlib.suppress(AttributeError):
                        delattr(cls, name)
        for name, value in self._cls_dict.items():
            setattr(cls, name, value)
        if not self._wrote_own_setattr and getattr(cls, '__attrs_own_setattr__', False):
            cls.__attrs_own_setattr__ = False
            if not self._has_custom_setattr:
                cls.__setattr__ = _OBJ_SETATTR
        return cls

    def _create_slots_class(self):
        """
        Build and return a new class with a `__slots__` attribute.
        """
        cd = {k: v for k, v in self._cls_dict.items() if k not in (*tuple(self._attr_names), '__dict__', '__weakref__')}
        if hasattr(sys, '_clear_type_descriptors'):
            sys._clear_type_descriptors(self._cls)
        if not self._wrote_own_setattr:
            cd['__attrs_own_setattr__'] = False
            if not self._has_custom_setattr:
                for base_cls in self._cls.__bases__:
                    if base_cls.__dict__.get('__attrs_own_setattr__', False):
                        cd['__setattr__'] = _OBJ_SETATTR
                        break
        existing_slots = {}
        weakref_inherited = False
        for base_cls in self._cls.__mro__[1:-1]:
            if base_cls.__dict__.get('__weakref__', None) is not None:
                weakref_inherited = True
            existing_slots.update({name: getattr(base_cls, name) for name in getattr(base_cls, '__slots__', [])})
        base_names = set(self._base_names)
        names = self._attr_names
        if self._weakref_slot and '__weakref__' not in getattr(self._cls, '__slots__', ()) and ('__weakref__' not in names) and (not weakref_inherited):
            names += ('__weakref__',)
        cached_properties = {name: cached_prop.func for name, cached_prop in cd.items() if isinstance(cached_prop, cached_property)}
        additional_closure_functions_to_update = []
        if cached_properties:
            class_annotations = _get_annotations(self._cls)
            for name, func in cached_properties.items():
                names += (name,)
                del cd[name]
                additional_closure_functions_to_update.append(func)
                annotation = inspect.signature(func).return_annotation
                if annotation is not inspect.Parameter.empty:
                    class_annotations[name] = annotation
            original_getattr = cd.get('__getattr__')
            if original_getattr is not None:
                additional_closure_functions_to_update.append(original_getattr)
            cd['__getattr__'] = _make_cached_property_getattr(cached_properties, original_getattr, self._cls)
        slot_names = [name for name in names if name not in base_names]
        reused_slots = {slot: slot_descriptor for slot, slot_descriptor in existing_slots.items() if slot in slot_names}
        slot_names = [name for name in slot_names if name not in reused_slots]
        cd.update(reused_slots)
        if self._cache_hash:
            slot_names.append(_HASH_CACHE_FIELD)
        cd['__slots__'] = tuple(slot_names)
        cd['__qualname__'] = self._cls.__qualname__
        cls = type(self._cls)(self._cls.__name__, self._cls.__bases__, cd)
        for item in itertools.chain(cls.__dict__.values(), additional_closure_functions_to_update):
            if isinstance(item, (classmethod, staticmethod)):
                closure_cells = getattr(item.__func__, '__closure__', None)
            elif isinstance(item, property):
                closure_cells = getattr(item.fget, '__closure__', None)
            else:
                closure_cells = getattr(item, '__closure__', None)
            if not closure_cells:
                continue
            for cell in closure_cells:
                try:
                    match = cell.cell_contents is self._cls
                except ValueError:
                    pass
                else:
                    if match:
                        cell.cell_contents = cls
        return cls

    def _make_getstate_setstate(self):
        """
        Create custom __setstate__ and __getstate__ methods.
        """
        state_attr_names = tuple((an for an in self._attr_names if an != '__weakref__'))

        def slots_getstate(self):
            """
            Automatically created by attrs.
            """
            return {name: getattr(self, name) for name in state_attr_names}
        hash_caching_enabled = self._cache_hash

        def slots_setstate(self, state):
            """
            Automatically created by attrs.
            """
            __bound_setattr = _OBJ_SETATTR.__get__(self)
            if isinstance(state, tuple):
                for name, value in zip(state_attr_names, state):
                    __bound_setattr(name, value)
            else:
                for name in state_attr_names:
                    if name in state:
                        __bound_setattr(name, state[name])
            if hash_caching_enabled:
                __bound_setattr(_HASH_CACHE_FIELD, None)
        return (slots_getstate, slots_setstate)

    def _add_method_dunders_unsafe(self, method: Callable) -> Callable:
        """
        Add __module__ and __qualname__ to a *method*.
        """
        method.__module__ = self._cls.__module__
        method.__qualname__ = f'{self._cls.__qualname__}.{method.__name__}'
        method.__doc__ = f'Method generated by attrs for class {self._cls.__qualname__}.'
        return method

    def _add_method_dunders_safe(self, method: Callable) -> Callable:
        """
        Add __module__ and __qualname__ to a *method* if possible.
        """
        with contextlib.suppress(AttributeError):
            method.__module__ = self._cls.__module__
        with contextlib.suppress(AttributeError):
            method.__qualname__ = f'{self._cls.__qualname__}.{method.__name__}'
        with contextlib.suppress(AttributeError):
            method.__doc__ = f'Method generated by attrs for class {self._cls.__qualname__}.'
        return method

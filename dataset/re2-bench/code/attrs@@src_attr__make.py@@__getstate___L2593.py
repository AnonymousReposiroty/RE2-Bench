import types

class Attribute:
    """
    *Read-only* representation of an attribute.

    .. warning::

       You should never instantiate this class yourself.

    The class has *all* arguments of `attr.ib` (except for ``factory`` which is
    only syntactic sugar for ``default=Factory(...)`` plus the following:

    - ``name`` (`str`): The name of the attribute.
    - ``alias`` (`str`): The __init__ parameter name of the attribute, after
      any explicit overrides and default private-attribute-name handling.
    - ``inherited`` (`bool`): Whether or not that attribute has been inherited
      from a base class.
    - ``eq_key`` and ``order_key`` (`typing.Callable` or `None`): The
      callables that are used for comparing and ordering objects by this
      attribute, respectively. These are set by passing a callable to
      `attr.ib`'s ``eq``, ``order``, or ``cmp`` arguments. See also
      :ref:`comparison customization <custom-comparison>`.

    Instances of this class are frequently used for introspection purposes
    like:

    - `fields` returns a tuple of them.
    - Validators get them passed as the first argument.
    - The :ref:`field transformer <transform-fields>` hook receives a list of
      them.
    - The ``alias`` property exposes the __init__ parameter name of the field,
      with any overrides and default private-attribute handling applied.


    .. versionadded:: 20.1.0 *inherited*
    .. versionadded:: 20.1.0 *on_setattr*
    .. versionchanged:: 20.2.0 *inherited* is not taken into account for
        equality checks and hashing anymore.
    .. versionadded:: 21.1.0 *eq_key* and *order_key*
    .. versionadded:: 22.2.0 *alias*

    For the full version history of the fields, see `attr.ib`.
    """
    __slots__ = ('name', 'default', 'validator', 'repr', 'eq', 'eq_key', 'order', 'order_key', 'hash', 'init', 'metadata', 'type', 'converter', 'kw_only', 'inherited', 'on_setattr', 'alias')

    def __init__(self, name, default, validator, repr, cmp, hash, init, inherited, metadata=None, type=None, converter=None, kw_only=False, eq=None, eq_key=None, order=None, order_key=None, on_setattr=None, alias=None):
        eq, eq_key, order, order_key = _determine_attrib_eq_order(cmp, eq_key or eq, order_key or order, True)
        bound_setattr = _OBJ_SETATTR.__get__(self)
        bound_setattr('name', name)
        bound_setattr('default', default)
        bound_setattr('validator', validator)
        bound_setattr('repr', repr)
        bound_setattr('eq', eq)
        bound_setattr('eq_key', eq_key)
        bound_setattr('order', order)
        bound_setattr('order_key', order_key)
        bound_setattr('hash', hash)
        bound_setattr('init', init)
        bound_setattr('converter', converter)
        bound_setattr('metadata', types.MappingProxyType(dict(metadata)) if metadata else _EMPTY_METADATA_SINGLETON)
        bound_setattr('type', type)
        bound_setattr('kw_only', kw_only)
        bound_setattr('inherited', inherited)
        bound_setattr('on_setattr', on_setattr)
        bound_setattr('alias', alias)

    def __getstate__(self):
        """
        Play nice with pickle.
        """
        return tuple((getattr(self, name) if name != 'metadata' else dict(self.metadata) for name in self.__slots__))

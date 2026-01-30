from decimal import (
    Clamped,
    Context,
    Decimal,
    Inexact,
    Overflow,
    Rounded,
    Underflow,
)
from boto3.compat import collections_abc

class TypeSerializer:
    """This class serializes Python data types to DynamoDB types."""

    def serialize(self, value):
        """The method to serialize the Python data types.

        :param value: A python value to be serialized to DynamoDB. Here are
            the various conversions:

            Python                                  DynamoDB
            ------                                  --------
            None                                    {'NULL': True}
            True/False                              {'BOOL': True/False}
            int/Decimal                             {'N': str(value)}
            string                                  {'S': string}
            Binary/bytearray/bytes (py3 only)       {'B': bytes}
            set([int/Decimal])                      {'NS': [str(value)]}
            set([string])                           {'SS': [string])
            set([Binary/bytearray/bytes])           {'BS': [bytes]}
            list                                    {'L': list}
            dict                                    {'M': dict}

            For types that involve numbers, it is recommended that ``Decimal``
            objects are used to be able to round-trip the Python type.
            For types that involve binary, it is recommended that ``Binary``
            objects are used to be able to round-trip the Python type.

        :rtype: dict
        :returns: A dictionary that represents a dynamoDB data type. These
            dictionaries can be directly passed to botocore methods.
        """
        dynamodb_type = self._get_dynamodb_type(value)
        serializer = getattr(self, f'_serialize_{dynamodb_type}'.lower())
        return {dynamodb_type: serializer(value)}

    def _get_dynamodb_type(self, value):
        dynamodb_type = None
        if self._is_null(value):
            dynamodb_type = NULL
        elif self._is_boolean(value):
            dynamodb_type = BOOLEAN
        elif self._is_number(value):
            dynamodb_type = NUMBER
        elif self._is_string(value):
            dynamodb_type = STRING
        elif self._is_binary(value):
            dynamodb_type = BINARY
        elif self._is_type_set(value, self._is_number):
            dynamodb_type = NUMBER_SET
        elif self._is_type_set(value, self._is_string):
            dynamodb_type = STRING_SET
        elif self._is_type_set(value, self._is_binary):
            dynamodb_type = BINARY_SET
        elif self._is_map(value):
            dynamodb_type = MAP
        elif self._is_listlike(value):
            dynamodb_type = LIST
        else:
            msg = f'Unsupported type "{type(value)}" for value "{value}"'
            raise TypeError(msg)
        return dynamodb_type

    def _is_null(self, value):
        if value is None:
            return True
        return False

    def _is_boolean(self, value):
        if isinstance(value, bool):
            return True
        return False

    def _is_number(self, value):
        if isinstance(value, (int, Decimal)):
            return True
        elif isinstance(value, float):
            raise TypeError('Float types are not supported. Use Decimal types instead.')
        return False

    def _is_string(self, value):
        if isinstance(value, str):
            return True
        return False

    def _is_binary(self, value):
        if isinstance(value, (Binary, bytearray, bytes)):
            return True
        return False

    def _is_set(self, value):
        if isinstance(value, collections_abc.Set):
            return True
        return False

    def _is_type_set(self, value, type_validator):
        if self._is_set(value):
            if False not in map(type_validator, value):
                return True
        return False

    def _is_map(self, value):
        if isinstance(value, collections_abc.Mapping):
            return True
        return False

    def _is_listlike(self, value):
        if isinstance(value, (list, tuple)):
            return True
        return False

import dataclasses as _dataclasses
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    ClassVar,
    Generic,
    Literal,
    TypeVar,
    Union,
    cast,
)
from pydantic_core import CoreSchema, PydanticCustomError, SchemaSerializer, core_schema
from ._internal import _fields, _internal_dataclass, _utils, _validators
from .annotated_handlers import GetCoreSchemaHandler, GetJsonSchemaHandler
from .json_schema import JsonSchemaValue

@_dataclasses.dataclass(**_internal_dataclass.slots_true)
class EncodedStr:
    """A str type that is encoded and decoded using the specified encoder.

    `EncodedStr` needs an encoder that implements `EncoderProtocol` to operate.

    ```python
    from typing import Annotated

    from pydantic import BaseModel, EncodedStr, EncoderProtocol, ValidationError

    class MyEncoder(EncoderProtocol):
        @classmethod
        def decode(cls, data: bytes) -> bytes:
            if data == b'**undecodable**':
                raise ValueError('Cannot decode data')
            return data[13:]

        @classmethod
        def encode(cls, value: bytes) -> bytes:
            return b'**encoded**: ' + value

        @classmethod
        def get_json_format(cls) -> str:
            return 'my-encoder'

    MyEncodedStr = Annotated[str, EncodedStr(encoder=MyEncoder)]

    class Model(BaseModel):
        my_encoded_str: MyEncodedStr

    # Initialize the model with encoded data
    m = Model(my_encoded_str='**encoded**: some str')

    # Access decoded value
    print(m.my_encoded_str)
    #> some str

    # Serialize into the encoded form
    print(m.model_dump())
    #> {'my_encoded_str': '**encoded**: some str'}

    # Validate encoded data
    try:
        Model(my_encoded_str='**undecodable**')
    except ValidationError as e:
        print(e)
        '''
        1 validation error for Model
        my_encoded_str
          Value error, Cannot decode data [type=value_error, input_value='**undecodable**', input_type=str]
        '''
    ```
    """

    encoder: type[EncoderProtocol]

    def __get_pydantic_json_schema__(
        self, core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        field_schema = handler(core_schema)
        field_schema.update(type='string', format=self.encoder.get_json_format())
        return field_schema

    def __get_pydantic_core_schema__(self, source: type[Any], handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        schema = handler(source)
        _check_annotated_type(schema['type'], 'str', self.__class__.__name__)
        return core_schema.with_info_after_validator_function(
            function=self.decode_str,
            schema=schema,
            serialization=core_schema.plain_serializer_function_ser_schema(function=self.encode_str),
        )

    def decode_str(self, data: str, _: core_schema.ValidationInfo) -> str:
        """Decode the data using the specified encoder.

        Args:
            data: The data to decode.

        Returns:
            The decoded data.
        """
        return self.encoder.decode(data.encode()).decode()

    def encode_str(self, value: str) -> str:
        """Encode the data using the specified encoder.

        Args:
            value: The data to encode.

        Returns:
            The encoded data.
        """
        return self.encoder.encode(value.encode()).decode()  # noqa: UP008

    def __hash__(self) -> int:
        return hash(self.encoder)

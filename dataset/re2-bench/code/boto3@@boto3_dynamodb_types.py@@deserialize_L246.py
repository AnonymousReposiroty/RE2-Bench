class TypeDeserializer:
    """This class deserializes DynamoDB types to Python types."""

    def deserialize(self, value):
        """The method to deserialize the DynamoDB data types.

        :param value: A DynamoDB value to be deserialized to a pythonic value.
            Here are the various conversions:

            DynamoDB                                Python
            --------                                ------
            {'NULL': True}                          None
            {'BOOL': True/False}                    True/False
            {'N': str(value)}                       Decimal(str(value))
            {'S': string}                           string
            {'B': bytes}                            Binary(bytes)
            {'NS': [str(value)]}                    set([Decimal(str(value))])
            {'SS': [string]}                        set([string])
            {'BS': [bytes]}                         set([bytes])
            {'L': list}                             list
            {'M': dict}                             dict

        :returns: The pythonic value of the DynamoDB type.
        """
        if not value:
            raise TypeError('Value must be a nonempty dictionary whose key is a valid dynamodb type.')
        dynamodb_type = list(value.keys())[0]
        try:
            deserializer = getattr(self, f'_deserialize_{dynamodb_type}'.lower())
        except AttributeError:
            raise TypeError(f'Dynamodb type {dynamodb_type} is not supported')
        return deserializer(value[dynamodb_type])

class IndirectReference(IndirectReferenceTuple):

    def __eq__(self, other):
        return other.__class__ is self.__class__ and other.object_id == self.object_id and (other.generation == self.generation)

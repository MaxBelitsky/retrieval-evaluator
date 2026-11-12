from enum import Enum


class ExtendedEnum(Enum):

    @classmethod
    def values(cls):
        return list(map(lambda c: c.value, cls))

    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value


class EmbeddingTextType(str, ExtendedEnum):
    query = "query"
    passage = "passage"

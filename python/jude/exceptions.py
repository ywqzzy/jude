"""jude exceptions — aligned with DuckDB's exception hierarchy."""


class Error(Exception):
    pass


class DatabaseError(Error):
    pass


class DataError(DatabaseError):
    pass


class OperationalError(DatabaseError):
    pass


class ProgrammingError(DatabaseError):
    pass


class IntegrityError(DatabaseError):
    pass


class InternalError(DatabaseError):
    pass


class NotSupportedError(DatabaseError):
    pass


class BinderException(DatabaseError):
    pass


class CatalogException(DatabaseError):
    pass


class ConnectionException(DatabaseError):
    pass


class ConstraintException(DatabaseError):
    pass


class ConversionException(DatabaseError):
    pass


class DependencyException(DatabaseError):
    pass


class FatalException(DatabaseError):
    pass


class HTTPException(DatabaseError):
    pass


class IOException(DatabaseError):
    pass


class InternalException(DatabaseError):
    pass


class InterruptException(DatabaseError):
    pass


class InvalidInputException(DatabaseError):
    pass


class InvalidTypeException(DatabaseError):
    pass


class NotImplementedException(DatabaseError):
    pass


class OutOfMemoryException(DatabaseError):
    pass


class OutOfRangeException(DatabaseError):
    pass


class ParserException(DatabaseError):
    pass


class PermissionException(DatabaseError):
    pass


class SequenceException(DatabaseError):
    pass


class SerializationException(DatabaseError):
    pass


class SyntaxException(DatabaseError):
    pass


class TransactionException(DatabaseError):
    pass


class TypeMismatchException(DatabaseError):
    pass


__all__ = [
    "Error", "DatabaseError", "DataError", "OperationalError",
    "ProgrammingError", "IntegrityError", "InternalError",
    "NotSupportedError", "BinderException", "CatalogException",
    "ConnectionException", "ConstraintException", "ConversionException",
    "DependencyException", "FatalException", "HTTPException",
    "IOException", "InternalException", "InterruptException",
    "InvalidInputException", "InvalidTypeException", "NotImplementedException",
    "OutOfMemoryException", "OutOfRangeException", "ParserException",
    "PermissionException", "SequenceException", "SerializationException",
    "SyntaxException", "TransactionException", "TypeMismatchException",
]

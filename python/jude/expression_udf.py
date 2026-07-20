"""jude.expression_udf — UDF registration and decorators."""

from .jude import attach_function, detach_function
import functools

__all__ = ["attach_function", "detach_function", "func", "cls"]


def func(fn=None, *, return_dtype=None, name=None):
    """Decorator for scalar UDFs.

    Usage:
        @jude.func(return_dtype="VARCHAR")
        def my_udf(x: str) -> str:
            return x.upper()
    """
    if fn is None:
        def decorator(f):
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs)
            wrapper._jude_return_dtype = return_dtype
            wrapper._jude_name = name or f.__name__
            wrapper._jude_is_func = True
            return wrapper
        return decorator

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    wrapper._jude_return_dtype = return_dtype
    wrapper._jude_name = name or fn.__name__
    wrapper._jude_is_func = True
    return wrapper


def cls(cls_=None, *, actor_number=None, return_dtype=None, name=None, gpus=0):
    """Decorator for stateful row-oriented UDF classes.

    Usage:
        @jude.cls(actor_number=1, return_dtype="VARCHAR")
        class MyStatefulUDF:
            def __init__(self):
                self.state = 0
            def __call__(self, x: str) -> str:
                self.state += 1
                return f"{self.state}: {x}"
    """
    if cls_ is None:
        def decorator(c):
            c._jude_return_dtype = return_dtype
            c._jude_name = name or c.__name__
            c._jude_actor_number = actor_number or 1
            c._jude_gpus = gpus
            c._jude_is_cls = True
            return c
        return decorator

    cls_._jude_return_dtype = return_dtype
    cls_._jude_name = name or cls_.__name__
    cls_._jude_actor_number = actor_number or 1
    cls_._jude_gpus = gpus
    cls_._jude_is_cls = True
    return cls_


cls.batch = None  # placeholder, will be set below


def cls_batch(cls_=None, *, actor_number=None, schema=None, name=None, batch_size=None, row_preserving=False, gpus=0):
    """Decorator for batch UDF classes.

    Usage:
        @jude.cls.batch(actor_number=1, schema={"result": "VARCHAR"})
        class MyBatchUDF:
            def __call__(self, table):
                import pyarrow as pa
                return pa.table({"result": ["processed"] * table.num_rows})
    """
    if cls_ is None:
        def decorator(c):
            c._jude_schema = schema
            c._jude_name = name or c.__name__
            c._jude_actor_number = actor_number or 1
            c._jude_batch_size = batch_size
            c._jude_row_preserving = row_preserving
            c._jude_gpus = gpus
            c._jude_is_cls_batch = True
            return c
        return decorator

    cls_._jude_schema = schema
    cls_._jude_name = name or cls_.__name__
    cls_._jude_actor_number = actor_number or 1
    cls_._jude_batch_size = batch_size
    cls_._jude_row_preserving = row_preserving
    cls_._jude_gpus = gpus
    cls_._jude_is_cls_batch = True
    return cls_


cls.batch = cls_batch

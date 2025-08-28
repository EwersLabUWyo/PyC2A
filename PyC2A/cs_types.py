import numpy as np
from numpy import frombuffer
from collections.abc import Callable
from pandas import Timestamp, Timedelta
from typing import Literal
from io import BufferedReader
from pandas import DataFrame
from functools import partial

from .file_handler import CampbellFile


class NSEC:
    name = "NSEC"
    size = 8
    return_type = Timestamp
    @staticmethod
    def from_bytes(self, b: bytes) -> Timestamp:
        S = int.from_bytes(b[:4], "little", signed=False)
        NS = int.from_bytes(b[4:8], "little", signed=False)
        total = np.int64(S)*np.int64(1_000_000_000) + np.int64(NS)//1e6*1e6
        return Timestamp("1990-01-01") + Timedelta(total, unit="ns")
class FP2:
    name = "FP2"
    size = 2
    return_type = np.dtype(">f2")
    @staticmethod
    def from_bytes(b: bytes) -> np.float16:
        # Bit 16: Sign, 0 = positive, 1 = negative
        # Bits 15, 14: Exponent, magnitude of negative decimal exponent
        # Bits 13-0: Magnitude of mantissa
        # +INF: sign = 0, mantissa = 8191
        # -INF: sign = 1, mantissa = 8191
        # NAN: sign = 1, mantissa = 8190
        tmp = int.from_bytes(b, byteorder="little", signed=False)
        S = tmp >> 15
        E = (tmp & 0x6000) >> 13
        M = (tmp & 0x1fff)

        match S, E, M:
            case 0, 0, 8191:
                return np.dtype(">f2")(np.inf)
            case 1, 0, 8191:
                return np.dtype(">f2")(-np.inf)
            case 1, 0, 8190:
                return np.dtype(">f2")(np.nan)
            case _:
                return np.dtype(">f2")((1 - 2*S)*M*10**(-E))

def handle_string_type(cf:CampbellFile):
    ascii_dtypes = {}
    for name in cf.file_dtypes:
        if "ASCII" in name:
            # written as 'ASCII(size)'
            size = name.split("(")[-2]
            ascii_dtypes[name] = np.dtype(f"|S{size}")
    np_readable_type_registry.update(ascii_dtypes)
    dtype_registry.update(ascii_dtypes)

# np.dtype("f4").itemsize
np_readable_type_registry = {
    "IEEE4": np.dtype(">f4"),
    "IEEE4B": np.dtype(">f4"),
    "IEEE8": np.dtype(">f8"),
    "IEEE8B": np.dtype(">f8"),
    "Long": np.dtype(">i4"),
    "UINT1": np.dtype(">ui1"),
    "UINT1B": np.dtype(">ui1"),
    "UINT2": np.dtype(">ui2"),
    "UINT2B": np.dtype(">ui2"),
    "UINT4": np.dtype(">ui4"),
    "UINT4B": np.dtype(">ui4"),
    "Bool8": np.dtype(">ui1"),
    "Bool8B": np.dtype(">ui1"),
    "ULONG": np.dtype(">ui4"),
    "LONG": np.dtype(">i4"),
    "Boolean": np.dtype("|b1"),
}
proprietary_type_registry = {
    "NSEC": NSEC,
    "SecNano": NSEC,
    "FP2": FP2,
}
dtype_registry = {}
dtype_registry.update(np_readable_type_registry).update(proprietary_type_registry)

#### vector/nonvector data parsing functions ####
#### handle data parsing ####
def data_parser_factory(csfile:CampbellFile) -> Callable:
    """creates a custom function to read a dataline, vectorizing as much of the computation as possible"""

    is_np_readable = tuple(d in np_readable_type_registry for d in csfile.file_dtypes)
    if sum(is_np_readable) == len(is_np_readable):
        return partial(vector_parser, csfile=csfile)
    
    parser_lst = []
    return_dtypes = []
    for i, name in enumerate(csfile.file_dtypes):
        if is_np_readable[i]:
            parser = partial(np.frombuffer, dtype=np_readable_type_registry[name])
            return_dtype = np_readable_type_registry[name]
        else:
            parser = proprietary_type_registry[name].from_bytes
            return_dtype = proprietary_type_registry[name].return_type
        parser_lst.append(parser)
        return_dtypes.append(return_dtype)

    def nonvector_parser(f: BufferedReader) -> np.ndarray:
        df = DataFrame({name:np.empty(csfile._frame_nrows, dtype=d) for name, d in zip(csfile.file_fieldnames, return_dtypes)})
        for r in df.index:
            for name, s, parser in zip(csfile.file_fieldnames, csfile._strides, parser_lst):
                df.loc[r, name] = parser(f.read(s))
        return df
    return nonvector_parser

def vector_parser(csfile: CampbellFile, f: BufferedReader) -> DataFrame:
    # e.g. [("IEEE4B", np.dtype(">f4")), ("Bool8", np.dtype(">ui1")), ("ULONG", np.dtype(">ui4"))]
    dtype = np.dtype([(k, np_readable_type_registry[k]) for k in csfile.file_dtypes])
    try:
        data_bytes = f.read(csfile._frame_data_size)
        if data_bytes == b'': 
            raise EOFError
        data = np.frombuffer(data_bytes, dtype=dtype).reshape(-1).tolist()
        return DataFrame(data=data, columns=csfile.file_fieldnames)
    except Exception:  # TODO: figure out what error numpy throws
        pass

# def nonvector_parser(csfile: CampbellFile, f: BufferedReader) -> DataFrame:
#     columns = {k:np.empty(csfile.frame_nrows, dtype=d) for k, d in zip(csfile.file_fieldnames, csfile.registered_dtypes)}
#     for r in range(csfile.frame_nrows):
#         for col, d, s, in zip(columns, csfile.registered_dtypes, csfile.strides):
#             data_bytes = f.read(s)
#             if data_bytes == b"":
#                 raise EOFError
#             #### TODO: everything should work now, EXCEPT THIS PART
#             columns[col][r] = parse_value(data_bytes, dtype=d)
#     return DataFrame(columns)

# data_parser_registry = (nonvector_parser, vector_parser)

# # ascii handled differently
# def parse_IEEE4(b:bytes) -> np.float32:
#     return frombuffer(b, dtype=befloat32)[0]
# def parse_IEEE8(b:bytes) -> np.float64:
#     return frombuffer(b, dtype=befloat64)[0]
# def parse_Long(b:bytes) -> np.int32:
#     return frombuffer(b, dtype=beint32)[0]
# def parse_UINT1(b:bytes) -> np.uint8:
#     return frombuffer(b, dtype=beuint8)[0]
# def parse_UINT2(b:bytes) -> np.uint16:
#     return frombuffer(b, dtype=beuint16)[0]
# def parse_UINT4(b:bytes) -> np.uint32:
#     return frombuffer(b, dtype=beuint32)[0]
# def parse_Bool8(b:bytes) -> np.uint8:
#     return frombuffer(b, dtype=beuint8)[0]

# def parse_Boolean(b:bytes) -> np.uint8:
#     # 1-byte boolean, stores a single value
#     return bool(frombuffer(b, dtype=beuint8))

# def parse_FP2(b:bytes) -> np.float16:
#     # Bit 16: Sign, 0 = positive, 1 = negative
#     # Bits 15, 14: Exponent, magnitude of negative decimal exponent
#     # Bits 13-0: Magnitude of mantissa
#     # +INF: sign = 0, mantissa = 8191
#     # -INF: sign = 1, mantissa = 8191
#     # NAN: sign = 1, mantissa = 8190
#     tmp = int.from_bytes(b, byteorder="little", signed=False)
#     S = tmp >> 15
#     E = (tmp & 0x6000) >> 13
#     M = (tmp & 0x1fff)

#     match S, E, M:
#         case 0, 0, 8191:
#             return np.dtype(">f16")(np.inf)
#         case 1, 0, 8191:
#             return np.dtype(">f16")(-np.inf)
#         case 1, 0, 8190:
#             return np.dtype(">f16")(np.nan)
#         case _:
#             return np.dtype(">f16")((1 - 2*S)*M*10**(-E))
        
# def parse_NSEC(b:bytes) -> Timestamp:
#     # 64bit Time stamp, divided as 4 bytes of seconds since 1990-01-01 00:00:00 and 4 bytes of naonoseconds into the second.
#     S = int.from_bytes(b[:4], byteorder="little", signed=False)#[::-1])
#     NS = int.from_bytes(b[4:8], byteorder="little", signed=False)#[::-1])
#     total = np.int64(S)*np.int64(1_000_000_000) + np.int64(NS)//1e6*1e6
#     return Timestamp("1990-01-01") + Timedelta(total, unit="ns")

# def parse_String(b:bytes) -> str:
#     raise NotImplementedError("String parsing has not been implemented yet.")

# dtype_process_func_registry: dict[str, Callable] = {
#     "IEEE4": parse_IEEE4,
#     "IEEE8": parse_IEEE8,
#     "Long": parse_Long,
#     "UINT1": parse_UINT1,
#     "UINT2": parse_UINT2,
#     "UINT4": parse_UINT4,
#     "Bool8": parse_Bool8,
#     "Boolean": parse_Boolean,
#     "FP2": parse_FP2,
#     "NSEC": parse_NSEC,
#     "String": parse_String,
#     "ULONG": parse_UINT4,
#     "LONG": parse_Long,
#     "SecNano": parse_NSEC,
#     "ASCII": parse_String,
# }
# dtype_size_registry: dict[str, int] = {
#     "IEEE4": 4,
#     "IEEE4": 4,
#     "IEEE8": 8,
#     "Long": 4,
#     "UINT1": 1,
#     "UINT2": 2,
#     "UINT4": 4,
#     "Bool8": 1,
#     "Boolean": 1,
#     "FP2": 2,
#     "NSEC": 8,
#     "String": -1,

#     "ULONG": 4,
#     "LONG": 4,
#     "SecNano": 8,
#     "ASCII": -1,
# }

# def parse_value(b:bytes, dtype:Literal["IEEE4", "IEEE8", "Long", "UINT1", "UINT2", "UINT4", "Bool8", "Boolean", "FP2", "NSEC", "String",]):
#     return dtype_process_func_registry[dtype](b)
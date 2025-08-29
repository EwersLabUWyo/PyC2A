# -----------------------------------------------------------------------------
#  ftype_specifics.py
#
#  module to binary data interpretation
#
#  Author: Alexander S Fox
#  Contact: https://www.afox.land   (replace with your preferred contact)
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
# -----------------------------------------------------------------------------

import numpy as np
from numpy import frombuffer
from collections.abc import Callable
from pandas import Timestamp, Timedelta
from typing import Literal
from io import BufferedReader
from pandas import DataFrame
from functools import partial


class NSEC:
    name = "NSEC"
    itemsize = 8
    return_type = Timestamp
    @staticmethod
    def from_bytes(b: bytes) -> Timestamp:
        S = int.from_bytes(b[:4], "little", signed=False)
        NS = int.from_bytes(b[4:8], "little", signed=False)
        total = np.int64(S)*np.int64(1_000_000_000) + np.int64(NS)//1e6*1e6
        return Timestamp("1990-01-01") + Timedelta(total, unit="ns")
class FP2:
    name = "FP2"
    itemsize = 2
    return_type = np.dtype(">f2")
    @staticmethod
    def from_bytes(b: bytes) -> np.float16:
        # Bit 16: Sign, 0 = positive, 1 = negative
        # Bits 15, 14: Exponent, magnitude of negative decimal exponent
        # Bits 13-0: Magnitude of mantissa
        # +INF: sign = 0, mantissa = 8191
        # -INF: sign = 1, mantissa = 8191
        # NAN: sign = 1, mantissa = 8190
        tmp = int.from_bytes(b, byteorder="big", signed=False)
        S = tmp >> 15
        E = (tmp & 0x6000) >> 13
        M = (tmp & 0x1fff)

        match S, E, M:
            case 0, 0, 8191:
                return np.float16(np.inf)
            case 1, 0, 8191:
                return np.float16(-np.inf)
            case 1, 0, 8190:
                return np.float16(np.nan)
            case _:
                return np.float16((1 - 2*S)*M*10**(-E))

def handle_string_type(cf):
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
    "UINT1": np.dtype(">u1"),
    "UINT1B": np.dtype(">u1"),
    "UINT2": np.dtype(">u2"),
    "UINT2B": np.dtype(">u2"),
    "UINT4": np.dtype(">u4"),
    "UINT4B": np.dtype(">u4"),
    "Bool8": np.dtype(">u1"),
    "Bool8B": np.dtype(">u1"),
    "ULONG": np.dtype(">u4"),
    "LONG": np.dtype(">i4"),
    "Boolean": np.dtype("|b1"),
}
proprietary_type_registry = {
    "NSEC": NSEC,
    "SecNano": NSEC,
    "FP2": FP2,
}
dtype_registry = dict()
dtype_registry.update(np_readable_type_registry)
dtype_registry.update(proprietary_type_registry)

#### vector/nonvector data parsing functions ####
#### handle data parsing ####
def data_parser_factory(csfile) -> Callable:
    """creates a custom function to read a dataline, vectorizing as much of the computation as possible"""

    is_np_readable = tuple(d in np_readable_type_registry for d in csfile.file_dtypes)
    if sum(is_np_readable) == len(is_np_readable):
        return lambda f: vector_parser(csfile, f)#partial(vector_parser, csfile=csfile)
    
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
        column_data = {name:np.empty(csfile._frame_nrows, dtype=d) for name, d in zip(csfile.file_fieldnames, return_dtypes)}
        for r in range(csfile._frame_nrows):
            for name, s, parser in zip(csfile.file_fieldnames, csfile._strides, parser_lst):
                column_data[name][r] = parser(f.read(s))
        return column_data
    return nonvector_parser

def vector_parser(csfile, f: BufferedReader) -> DataFrame:
    # e.g. [("IEEE4B", np.dtype(">f4")), ("Bool8", np.dtype(">ui1")), ("ULONG", np.dtype(">ui4"))]
    dtype = np.dtype([(f"{k}_{i}", np_readable_type_registry[k]) for i, k in enumerate(csfile.file_dtypes)])
    try:
        data_bytes = f.read(csfile._frame_data_size)
        if data_bytes == b'': 
            raise EOFError
        data = np.frombuffer(data_bytes, dtype=dtype).reshape(-1).tolist()
        #### TODO: fix hacky solution that i implemented to handle making this method compatible with nonvector_parser
        column_data = {name: data[:, i] for i, name in enumerate(csfile.file_fieldnames)}
        return column_data
    except Exception:  # TODO: figure out what error numpy throws
        pass
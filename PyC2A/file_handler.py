from dataclasses import dataclass
from typing import Literal, Callable, Any
from inspect import signature
from functools import partial
from io import BufferedReader

import numpy as np
from pandas import Timestamp, Timedelta, DataFrame

from .cs_types import *

#### shared parsing helpers ####
def parse_ascii_header_line(ln):
    return ln.decode("ascii").replace("\"", "").strip().split(",")

#### format handler classes ####
class TOB1:
    header_size = 8
    footer_size = 4
    @staticmethod
    def parse_header(b: bytes) -> tuple[Timestamp, None]:
        raise NotImplementedError("TOB1 parsing not implemented yet")
    @staticmethod
    def parse_footer(b: bytes) -> None:
        raise NotImplementedError("TOB1 parsing not implemented yet")

class TOB2:
    header_size = 8
    footer_size = 4
    @staticmethod
    def parse_header(b: bytes) -> tuple[Timestamp, None]:
        return dtype_registry["NSEC"].from_bytes(b), None
    @staticmethod
    def parse_footer(b: bytes) -> None:
        return None

class TOB3:
    header_size = 12
    footer_size = 4
    @staticmethod
    def parse_header(b: bytes) -> tuple[Timestamp, int]:
        nsec_bytes = b[:8]
        recnum_bytes = b[8:]
        return dtype_registry["NSEC"].from_bytes(nsec_bytes), dtype_registry["UINT4"](recnum_bytes)
    @staticmethod
    def parse_footer(b: bytes) -> None:
        return None


class TOA5:
    header_size = 0
    footer_size = 0
    @staticmethod
    def parse_header(b: bytes) -> tuple[None, None]:
        return None, None
    @staticmethod
    def parse_footer(b: bytes) -> None:
        return None

format_registry: dict[str, Any] = {
    "TOB1": TOB1,
    "TOB2": TOB2,
    "TOB3": TOB3,
    "TOA5": TOA5,
}

#### main file class ####
@dataclass
class CampbellFile:
    fmt: Literal["TOB1", "TOB2", "TOB3", "TOA5"]
    station: str = None
    model: str = None
    serial_number: str = None
    os_version: str = None
    program: str = None
    signature: str = None
    created: Timestamp = None
    table: str = None
    interval: int = None
    frame_size: int = None
    intended_table_size: int = None
    validation: int = None
    frame_time_res: str = None
    file_fieldnames: tuple[str] = None
    file_units: tuple[str] = None
    file_process: tuple[str] = None
    file_dtypes: tuple[str] = None

    def __post_init__(self):
        self.registered_dtypes = tuple(dtype_registry[name] for name in self.file_dtypes)
        self.strides = tuple(rdt.itemsize for rdt in self.registered_dtypes)

        self.frame_data_size = self.frame_size - self.handler.frame_header_size - self.handler.frame_footer_size
        self.frame_nrows = self.frame_data_size // sum(self.strides)

        # frame of the file is readable by np.frombuffer if all datatypes are "native" numpy types (much faster)
        n_dtypes = len(set(self.file_dtypes))
        self.is_vector_readable = sum(d in np_readable_type_registry for d in self.file_dtypes) == n_dtypes

    @property
    def handler(self):
        return format_registry[self.fmt]
    
    @property
    def data_parser(self):
        return partial(data_parser_registry[self.is_vector_readable], csfile=self) 

    def parse_frame_header(self, f) -> tuple[Timestamp, int]:
        size = self.handler.header_size
        b = f.read(size)
        if b == b"":
            raise EOFError
        return self.handler.parse_header(b)

    def parse_frame_footer(self, f) -> Any:
        size = self.handler.footer_size
        b = f.read(size)
        if b == b"":
            raise EOFError
        return self.handler.parse_footer(b)
    

    # def parse_frame_data(self, f) -> DataFrame:
    #     # only one dtype, can process all at once
    #     if self.n_dtypes == 1:
    #         try:
    #             data_bytes = f.read(self.frame_data_size)
    #             if data_bytes == b'': 
    #                 raise EOFError
    #             dtype = self.registered_dtypes[0]
    #             data = np.frombuffer(data_bytes, dtype=dtype).reshape(-1, len(self.file_fieldnames))
    #             return DataFrame(data=data, columns=self.file_fieldnames)
    #         except Exception:
    #             pass
        
    #     # multiple datatypes
    #     columns = {k:np.empty(self.frame_nrows, dtype=d) for k, d in zip(self.file_fieldnames, self.registered_dtypes)}
    #     for r in range(self.frame_nrows):
    #         for col, d, s, in zip(columns, self.registered_dtypes, self.strides):
    #             data_bytes = f.read(s)
    #             if data_bytes == b"":
    #                 raise EOFError
    #             columns[col][r] = parse_value(data_bytes, dtype=d)
    #     return DataFrame(columns)
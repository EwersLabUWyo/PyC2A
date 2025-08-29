# -----------------------------------------------------------------------------
#  ftype_specifics.py
#
#  module to handle differences between TOB1/2/3 and to handle the main processing methods
#
#  Author: Alexander S Fox
#  Contact: https://www.afox.land   (replace with your preferred contact)
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
# -----------------------------------------------------------------------------

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
        return dtype_registry["NSEC"].from_bytes(nsec_bytes), np.frombuffer(recnum_bytes, dtype_registry["UINT4"])
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
    """Dataclass containing raw file metadata, plus some generic methods to handle binary data processing and to handle multiple TOB file types."""
    fmt: Literal["TOB1", "TOB2", "TOB3", "TOA5"] = None
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

    def manual_post_init(self):
        # instantiate information not found in the raw file metadata
        self._handler = format_registry[self.fmt]
        self._registered_dtypes = tuple(dtype_registry[name] for name in self.file_dtypes)
        strides = []
        self._strides = tuple(rdt.itemsize for rdt in self._registered_dtypes)

        self._frame_data_size = self.frame_size - self.handler.header_size - self.handler.footer_size
        self._frame_nrows = self._frame_data_size // sum(self._strides)
        self._nframes = self.intended_table_size // self._frame_nrows

        self._data_parser = data_parser_factory(self)


    @property
    def handler(self):
        return self._handler

    def parse_frame_header(self, f) -> tuple[Timestamp, int]:
        size = self.handler.header_size
        b = f.read(size)
        if b == b"":
            raise EOFError
        return self.handler.parse_header(b)
    
    def parse_frame_data(self, f):
        return self._data_parser(f)
    
    def parse_frame_footer(self, f) -> Any:
        size = self.handler.footer_size
        b = f.read(size)
        if b == b"":
            raise EOFError
        return self.handler.parse_footer(b)
    
    def parse_whole_frame(self, f) -> tuple[Timestamp, int, DataFrame, Any]:
        t_start, recnum = self.parse_frame_header(f)
        df = self.parse_frame_data(f)
        footer = self.parse_frame_footer(f)
        return t_start, recnum, df, footer
    
def compile_to_dataframe(csfile:CampbellFile, all_data: list[dict[str, np.ndarray]]) -> DataFrame:
    columns = list(all_data[0].keys())
    columns.remove("TIMESTAMP")
    if "RECORD" in columns:
        columns.remove("RECORD")

    arrays = [
        np.concatenate([frame[col] for frame in all_data], axis=0)
        for col in columns
        if not (
            np.issubdtype(all_data[0][col].dtype, np.datetime64) or
            np.issubdtype(all_data[0][col].dtype, np.str_) or
            all_data[0][col].dtype == object  # covers generic Python strings
        )
    ]
    timestamps = np.concatenate([frame["TIMESTAMP"] for frame in all_data], axis=0)
    
    if "RECORD" in all_data[0]:
        records = np.concatenate([frame["RECORD"] for frame in all_data], axis=0)

    df = DataFrame(data=np.stack(arrays, axis=1), columns=columns)
    df["TIMESTAMP"] = timestamps
    if "RECORD" in all_data[0]:
        df["RECORD"] = records
    return df
# -----------------------------------------------------------------------------
#  ftype_specifics.py
#
#  module to handle differences between TOB1/2/3
#
#  Author: Alexander S Fox
#  Contact: https://www.afox.land   (replace with your preferred contact)
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
# -----------------------------------------------------------------------------
from typing import Literal
from io import BufferedReader
from inspect import signature

from numpy import int32
from pandas import DataFrame, Timestamp
from .cs_types import *

from dataclasses import dataclass
from typing import Literal
@dataclass
class CampbellFileMeta:
    fmt:Literal["TOB2", "TOB3", "TOB1", "TOA5"]=None
    station:str=None
    model:str=None
    serial_number:str=None
    os_version:str=None
    program:str=None
    signature:str=None
    created:Timestamp=None
    table:str=None
    interval:int=None
    frame_size:int=None
    intended_table_size:int=None
    validation:int=None
    frame_time_res:str=None
    file_fieldnames:tuple[str]=None
    file_units:tuple[str]=None
    file_process:tuple[str]=None
    file_dtype:tuple[str]=None

def parse_ascii_header_line(ln):
    return ln.decode("ascii").replace("\"", "").strip().split(",")

def parse_TOB1_frame_header(b:bytes) -> Timestamp:
    raise NotImplementedError("TOB1 processing is not implemented yet")

def parse_TOB2_frame_header(b:bytes) -> Timestamp:
    return parse_NSEC(b), None

def parse_TOB3_frame_header(b:bytes) -> tuple[Timestamp, int32]:
    # we parse bytes in big-endian format
    nsec_bytes = b[:8]
    recnum_bytes = b[8:]
    return parse_NSEC(nsec_bytes), parse_UINT4(recnum_bytes)

def parse_TOA5_frame_header(b:bytes) -> None:
    """included for completeness"""
    return None, None

def parse_frame_header(f:bytes, fmt:Literal["TOB1", "TOB2", "TOB3", "TOA5"]):
    header_bytes = f.read(frame_header_size_map[fmt])
    if header_bytes == "":
        raise EOFError
    
    match fmt:
        case "TOB1":
            return parse_TOB1_frame_header(header_bytes)
        case "TOB2":
            return parse_TOB2_frame_header(header_bytes)
        case "TOB3":
            return parse_TOB3_frame_header(header_bytes)
        case "TOA5":
            return parse_TOA5_frame_header(header_bytes)
        
frame_header_size_map = {
    "TOB1": None,
    "TOB2": 8,
    "TOB3": 12,
    "TOA5": 0,
}

def parse_frame_footer(f:bytes, fmt:Literal["TOB1", "TOB2", "TOB3", "TOA5"]):
    """Included for completeness"""
    footer_bytes = f.read(frame_footer_size_map[fmt])
    if footer_bytes == "":
        raise EOFError
    
    return None

frame_footer_size_map = {
    "TOB1": None,
    "TOB2": 4,
    "TOB3": 4,
    "TOA5": 0,
}

def parse_frame_data(f:BufferedReader, fileinfo:CampbellFileMeta, frame_data_size:int) -> DataFrame:
    # only one dtype, can process all at once
    if len(set(fileinfo.file_dtype)) == 1:
        dtype = signature(dtype_process_func_registry[fileinfo.file_dtype[0]]).return_annotation
        data_bytes = f.read(frame_data_size)
        
        if data_bytes == '': 
            raise EOFError
        
        data = np.frombuffer(data_bytes, dtype=dtype).reshape(-1, len(fileinfo.file_fieldnames))
        return DataFrame(data=data, columns=fileinfo.file_fieldnames)
    
    # multiple datatypes
    strides = tuple(dtype_size_registry[d] for d in fileinfo.file_dtype)
    nrows = frame_data_size // sum(strides)
    columns = {
        k:np.empty(nrows, dtype=signature(dtype_process_func_registry[d]).return_annotation)
        for k, d in zip(fileinfo.file_fieldnames, fileinfo.file_dtype)
    }
    for r in range(nrows):
        for col, d, s, in zip(columns, fileinfo.file_dtype, strides):
            data_bytes = f.read(s)
            
            if data_bytes == "":
                raise EOFError
            
            columns[col][r] = parse_value(data_bytes, dtype=d)
    return DataFrame(columns)
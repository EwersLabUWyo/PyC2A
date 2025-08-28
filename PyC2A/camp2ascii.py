# -----------------------------------------------------------------------------
#  camp2ascii.py
#
#  A program to read TOB-format binary files into plaintext
#
#  Author: Alexander S Fox
#  Contact: https://www.afox.land   (replace with your preferred contact)
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
# -----------------------------------------------------------------------------

from pathlib import Path
import warnings
import numpy as np
from pandas import DataFrame, Timedelta, date_range, concat, read_csv
from tqdm import tqdm
from inspect import signature

from .file_handler import *
from .cs_types import *

def camp2ascii(fn:Path, chunksize:Timedelta=None) -> DataFrame:

    fileinfo = CampbellFile()

    with open(fn, "rb") as f:
        # parse header
        # example TOB3 Header
        """
        "TOB3","2991","CR6","2991","CR6.Std.04","CPU:TEST.EC.v18.CR6","52714","2018-06-08 00:00:00"
        "ts_data","100 MSEC","984","950400","26624","Sec100Usec","           0","           0","0730014788"
        "Ux","Uy","Uz","Ts","diag_sonic","H2O","CO2","amb_press","diag_irga","CO2_u_mol","H2O_m_mol"
        "","","","degC","","mg/m^3","g/m^3","kPa","unitless","umol/mol","mmol/mol"
        "Smp","Smp","Smp","Smp","Smp","Smp","Smp","Smp","Smp","Smp","Smp"
        "IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B" 
        """
        (
            fileinfo.fmt,
            fileinfo.station,
            fileinfo.model,
            fileinfo.serial_number,
            fileinfo.os_version,
            fileinfo.program,
            fileinfo.signature,
            fileinfo.created,
            *_
        ) = parse_ascii_header_line(next(f))
        (
            fileinfo.table,
            fileinfo.interval,
            fileinfo.frame_size,
            fileinfo.intended_table_size,
            fileinfo.validation,
            fileinfo.frame_time_res,
            *_ 
        ) = parse_ascii_header_line(next(f))
        fileinfo.frame_size = int(fileinfo.frame_size)
        fileinfo.intended_table_size = int(fileinfo.intended_table_size)

        fileinfo.file_fieldnames = tuple(next(f).decode("ascii").replace("\"", "").strip().split(","))
        fileinfo.file_units = tuple(parse_ascii_header_line(next(f)))
        fileinfo.file_process = tuple(parse_ascii_header_line(next(f)))
        fileinfo.file_dtype = parse_ascii_header_line(next(f))
        handle_string_type(fileinfo)
            
        #### parse the data ####
        if fileinfo.fmt == "TOA5":
            return fileinfo, read_csv(fn, skiprows=[0, 2, 3], na_values=["-9999", "NAN"], parse_dates=["TIMESTAMP"])

        frame_header_size = frame_header_size_map[fileinfo.fmt]
        frame_footer_size = frame_footer_size_map[fileinfo.fmt]
        frame_data_size = fileinfo.frame_size - frame_header_size - frame_footer_size

        camp2timedelta = {
            "MSEC": "ms"
        }
        dt, dt_unit = fileinfo.interval.split(" ")
        dt_unit = camp2timedelta[dt_unit]
        dt = Timedelta(dt + dt_unit)
        if chunksize is not None:
            records_per_chunk = chunksize // dt

        dfs = []
        # for computing clock drift: we keep an independant clock counter to avoid datalogger clock drift.
        nrows_expected = frame_data_size // sum(dtype_size_registry[d] for d in fileinfo.file_dtype)
        expected_dt_per_frame = nrows_expected*dt
        read_first_frame = False
        try:
            while True:
                if read_first_frame:
                    _, recnum_start = parse_frame_header(f, fileinfo.fmt)
                    t_start += expected_dt_per_frame
                else:
                    t_start, recnum_start = parse_frame_header(f, fileinfo.fmt)
                    read_first_frame = True

                frame_data = parse_frame_data(f, fileinfo, frame_data_size)
                _ = parse_frame_footer(f, fileinfo.fmt)
                
                # address possible clock drift
                if frame_data.shape[0] != nrows_expected:
                    msg = f"Number of frame rows ({frame_data.shape[0]}) different than expected ({nrows_expected}). Defaulting to datalogger clock at {t_start}."
                    warnings.warn(msg)
                    t_start = t_start.round(freq=dt)
                
                # format data into a dataframe
                frame_data["TIMESTAMP"] = date_range(t_start, freq=dt, periods=frame_data.shape[0])
                if recnum_start is not None:
                    frame_data["RECORD"] = np.arange(recnum_start, recnum_start + frame_data.shape[0])
                frame_data = frame_data.sort_values("TIMESTAMP")
                dfs.append(frame_data)  # could speed this up a lot

                # yield data if the user requested the file to be chunked up
                if chunksize is not None:
                    if sum(df.shape[0] for df in dfs) >= records_per_chunk:
                        df = concat(dfs).sort_values("TIMESTAMP").reset_index(drop=True)            
                        yield fileinfo, df.iloc[:records_per_chunk]
                        dfs = [df.iloc[records_per_chunk:]]

        except (EOFError, IndexError):
            yield fileinfo, concat(dfs).sort_values("TIMESTAMP").reset_index(drop=True)

if __name__ == "__main__":
    chunksize = Timedelta("30min")
    fn = "/Users/alex/Documents/Work/UWyo/Workshops/EddyProConfigEditor/2991.ts_data_2598.dat"
    gen = camp2ascii(fn, chunksize)
    while True:
        _, data = next(gen)
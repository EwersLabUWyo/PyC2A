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
from tqdm import trange
from inspect import signature

from tqdm import tqdm
from .file_handler import *
from .cs_types import *

def camp2ascii(fn:Path, nlines=None) -> tuple[CampbellFile, DataFrame]:
    """Converts a campbell scientific TOB file to a dataframe. Returns a tuple of the CampbellFile dataclass (containing raw file metadata) and a pandas DataFrame of the raw data."""

    csfile = CampbellFile()

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
            csfile.fmt,
            csfile.station,
            csfile.model,
            csfile.serial_number,
            csfile.os_version,
            csfile.program,
            csfile.signature,
            csfile.created,
            *_
        ) = parse_ascii_header_line(next(f))
        (
            csfile.table,
            csfile.interval,
            csfile.frame_size,
            csfile.intended_table_size,
            csfile.validation,
            csfile.frame_time_res,
            *_ 
        ) = parse_ascii_header_line(next(f))
        csfile.frame_size = int(csfile.frame_size)
        csfile.intended_table_size = int(csfile.intended_table_size)

        csfile.file_fieldnames = tuple(next(f).decode("ascii").replace("\"", "").strip().split(","))
        csfile.file_units = tuple(parse_ascii_header_line(next(f)))
        csfile.file_process = tuple(parse_ascii_header_line(next(f)))
        csfile.file_dtypes = parse_ascii_header_line(next(f))
        handle_string_type(csfile)

        csfile.manual_post_init()
            
        #### parse the data ####
        if csfile.fmt == "TOA5":
            return csfile, read_csv(fn, skiprows=[0, 2, 3], na_values=["-9999", "NAN"], parse_dates=["TIMESTAMP"])

        ### TODO: improve time interval parsing.
        camp2timedelta = {
            "MSEC": "ms",
            "MIN": "min"
        }
        dt, dt_unit = csfile.interval.split(" ")
        dt_unit = camp2timedelta[dt_unit]
        dt = Timedelta(dt + dt_unit)

        expected_dt_per_frame = csfile._frame_nrows*dt
        t_start, recnum_start, df_dict, _ = csfile.parse_whole_frame(f)
        df_dict_template = df_dict.copy()
        try:
            if nlines is not None:
                pbar = trange(nlines // csfile._frame_nrows)
                frames = [df_dict_template]* (nlines // csfile._frame_nrows)
            else:
                pbar = trange(csfile._nframes)
                frames = [df_dict_template]* csfile._nframes
            for framenum in pbar:
                # candidate_t_start is only used if we lose track of the clock
                candidate_t_start, recnum_start, df_dict, _ = csfile.parse_whole_frame(f)
                t_start += expected_dt_per_frame

                try:
                    last_frame_t_start = frames[framenum - 1]["TIMESTAMP"][0]
                    if np.abs(candidate_t_start - last_frame_t_start) > expected_dt_per_frame*framenum*1.1:  # max acceptable drift of 10% per frame
                        msg = f"Unacceptable clock drift! Setting clock {last_frame_t_start} -> {candidate_t_start}"
                        warnings.warn(msg)
                        t_start = candidate_t_start
                except KeyError:
                    pass
                
                # format data into a dataframe
                df_dict["TIMESTAMP"] = date_range(t_start, freq=dt, periods=csfile._frame_nrows)
                if recnum_start is not None:
                    df_dict["RECORD"] = np.arange(recnum_start, recnum_start + csfile._frame_nrows)
                frames[framenum] = df_dict  # could speed this up a lot
        except (EOFError, IndexError):
            msg = f"EOFError! File {fn} may be corrupted. Outputting results anyway..."
            warnings.warn(msg)
            return csfile, compile_to_dataframe(csfile, frames)
        return csfile, compile_to_dataframe(csfile, frames)
    
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
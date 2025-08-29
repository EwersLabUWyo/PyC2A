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

def camp2ascii(fn:Path, chunksize:int=None, progress:bool=True) -> tuple[CampbellFile, DataFrame]:
    """
    Converts a Campbell Scientific TOB file to a DataFrame.
    If chunksize is given, yields (csfile, DataFrame) for each chunk of lines.
    Otherwise, returns (csfile, DataFrame) for the whole file. Pass progress=True for a progress bar display.
    """
    if chunksize is None:
        return next(_camp2ascii_gen(fn, chunksize=None, progress=progress))
    else:
        return _camp2ascii_gen(fn, chunksize=chunksize, progress=progress)
    
def _camp2ascii_gen(fn: Path, chunksize=None, progress=True):
    #### parse ascii header ####
    csfile = CampbellFile()
    with open(fn, "rb") as f:
        line1 = next(f)
        # special case of TOA5: just an ascii file. No need to do anything else.
        if "TOA5" in str(line1):
            df = read_csv(fn, skiprows=[0, 2, 3], na_values=["-9999", "NAN"], parse_dates=["TIMESTAMP"])
            if chunksize is None:
                return csfile, df
            else:
                for i in range(0, len(df), chunksize):
                    yield csfile, df.iloc[i:i+chunksize]
            return

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
        ) = parse_ascii_header_line(line1)
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

        camp2timedelta = {
            "MSEC": "ms",
            "MIN": "min"
        }
        dt, dt_unit = csfile.interval.split(" ")
        dt_unit = camp2timedelta[dt_unit]
        dt = Timedelta(dt + dt_unit)

        #### prep for main loop ####
        frames = []
        lines_in_chunk = 0
        total_lines = 0

        expected_dt_per_frame = csfile._frame_nrows * dt
        t_start, recnum_start, frame, _ = csfile.parse_whole_frame(f)
        frame["TIMESTAMP"] = date_range(t_start, freq=dt, periods=csfile._frame_nrows)
        if recnum_start is not None:
            frame["RECORD"] = np.arange(recnum_start, recnum_start + csfile._frame_nrows)
        frames.append(frame)
        lines_in_chunk += csfile._frame_nrows
        total_lines += csfile._frame_nrows

        try:
            max_frames = csfile._nframes
            pbar = range(max_frames)
            if progress:
                pbar = trange(max_frames)
            for framenum in pbar:
                candidate_t_start, recnum_start, frame, _ = csfile.parse_whole_frame(f)
                t_start += expected_dt_per_frame

                # try:
                #     last_frame_t_start = frames[-1]["TIMESTAMP"][0]
                #     if np.abs(candidate_t_start - last_frame_t_start) > expected_dt_per_frame * framenum * 1.1:
                #         msg = f"Unacceptable clock drift! Setting clock {last_frame_t_start} -> {candidate_t_start}"
                #         warnings.warn(msg)
                #         t_start = candidate_t_start
                # except (KeyError, IndexError):
                #     pass

                frame["TIMESTAMP"] = date_range(t_start, freq=dt, periods=csfile._frame_nrows)
                if recnum_start is not None:
                    frame["RECORD"] = np.arange(recnum_start, recnum_start + csfile._frame_nrows)
                frames.append(frame)
                lines_in_chunk += csfile._frame_nrows
                total_lines += csfile._frame_nrows

                if chunksize is not None and lines_in_chunk >= chunksize:
                    yield csfile, compile_to_dataframe(csfile, frames)
                    frames = []
                    lines_in_chunk = 0
                    if progress: pbar.update(lines_in_chunk)
                elif chunksize is None and progress:
                    pbar.update(1)

            # Yield any remaining frames at the end
            if frames:
                if progress: pbar.update(lines_in_chunk)
                yield csfile, compile_to_dataframe(csfile, frames)
        except (EOFError, IndexError):
            msg = f"EOFError! File {fn} may be corrupted. Outputting results anyway..."
            warnings.warn(msg)
            if frames:
                yield csfile, compile_to_dataframe(csfile, frames)
    return
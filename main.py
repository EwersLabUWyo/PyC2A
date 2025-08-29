# -----------------------------------------------------------------------------
#  camp2ascii.py
#
#  A program to read TOB-format binary files into plaintext. Main script 
#  for CLI usage.
#
#  Author: Alexander S Fox
#  Contact: https://www.afox.land   (replace with your preferred contact)
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from PyC2A.camp2ascii import camp2ascii
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="input binary file")
    parser.add_argument("-o", help="Output file. If --chunksize is provided, this should be a directory.")
    parser.add_argument("-c", "--chunksize", help="number of lines per chunk. Default None (parse entire dataframe)", default=None, type=int)
    parser.add_argument("--no-progress", action="store_true", help="whether to hide the progress bar")
    parser.add_argument("-of", "--oformat", help='Output format. Options are "ascii" (csv, default) or "feather"', default="ascii")

    args = parser.parse_args()
    
    res = camp2ascii(Path(args.i), chunksize=args.chunksize, progress=(not args.no_progress))
    
    if args.chunksize is not None:
        out_dir = Path(args.o)
        if not out_dir.exists() or not out_dir.is_dir():
            raise ValueError("When using --chunksize, -o must be an existing directory.")
        
        gen = camp2ascii(Path(args.i), chunksize=args.chunksize, progress=(not args.no_progress))
        for _, df in gen:
            ts = df["TIMESTAMP"].iloc[0]
            stem = ts.strftime(r'%Y-%m-%dT%H%M%S')
            match args.oformat:
                case "ascii":
                    df.to_csv(Path(args.o) / f"{stem}.csv", index=False)
                case "feather":
                    df.to_feather(Path(args.o) / f"{stem}.feather")
                case _:
                    msg = f"{args.oformat} is not a valid file format. Please choose one of 'feather', or 'ascii'."
                    raise NotImplementedError(msg)
    else:
        _, df = camp2ascii(Path(args.i), chunksize=args.chunksize, progress=(not args.no_progress))
        match args.oformat:
            case "ascii":
                df.to_csv(Path(args.o), index=False)
            case "feather":
                df.to_feather(Path(args.o))
            case _:
                msg = f"{args.oformat} is not a valid file format. Please choose one of 'feather', or 'ascii'."
                raise NotImplementedError(msg)

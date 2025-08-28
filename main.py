if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from PyC2A.camp2ascii import camp2ascii
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="input binary file")
    parser.add_argument("-o", help="output file")
    parser.add_argument("-of", "--oformat", help='Output format. Options are "ascii" (csv, default) or "feather"', default="ascii")

    args = parser.parse_args()
    args.oformat
    csfile, df = camp2ascii(Path(args.i))

    match args.oformat:
        case "ascii":
            df.to_csv(Path(args.o), index=False)
        case "feather":
            df.to_feather(Path(args.o))
    

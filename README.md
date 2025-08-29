# PyC2A
PyC2A is a simple python module for parsing Campbell Scientific TOB binary files to ASCII or other formats. Unlike CardConvert, this program is easier to automate and include in workflows, but is not as flexible, is slower, and doesn't deal well with corrupted files. Currently, only TOB3 and TOB2 are implemented. Also, it's not very fast, but I'm working on that part. [Mathias Bavay's camp2ascii](https://gitlabext.wsl.ch/bavay/camp2ascii/-/tree/master?ref_type=heads) program is much faster if you need something more performant. 

Dependencies are pandas, numpy, and tqdm (for progress bars).

# Usage
This module is not distributed as a package yet. To use it, place the PyC2A directory in your working directory. To run from the command line, place `main.py` in the parent directory of PyC2A, or run `python main.py -h` to see command line options. Example usage:

```zsh
% ls
LICENSE         PyC2A           README.md       main.py
% ls PyC2A
__init__.py             camp2ascii.py           file_handler.py
__pycache__             cs_types.py             ftype_specifics.py
% python main.py -i ts_data_2991.dat -o ts_data_2991.csv -f ascii
```

Or import the main parsing function into another script sitting in the parent directory of PyC2A. Example usage:

```python
from pathlib import Path
import matplotlib.pyplot as plt
from PyC2A.camp2ascii import camp2ascii
if __name__ == "__main__":
    fn = Path("ts_data_2991.dat")
    csfile, df = camp2ascii(fn)
    df.plot(figsize=(15, 5))
    plt.show()
```
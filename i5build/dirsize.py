"""Print the size in bytes of each directory given on the command line.

Lives on the remote machine so the caller only ever passes a bare path -- every attempt
to parse `dir` output across ssh ran into the remote console codepage turning the
summary line into mojibake.
"""
import os
import sys

for path in sys.argv[1:]:
    total = 0
    files = 0
    for dirpath, _, filenames in os.walk(path):
        for fn in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, fn))
                files += 1
            except OSError:
                pass
    print(f"{path}\t{total}\t{files}")

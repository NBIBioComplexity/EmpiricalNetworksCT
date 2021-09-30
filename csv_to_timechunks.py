import os
import sys
import pandas as pd
from datetime import datetime
import time
import pickle


if len(sys.argv) < 2:
    print("Error. No CSV input file supplied.")
    exit()
else:
    infile = sys.argv[1]


edgecsv = pd.read_csv(infile, sep=' ')
edgecsv = edgecsv.sort_values(by='timestamp', ascending=True)


def get_timelump(df, dt):
    # Generate dict with dataframe for each lump of time of length dt (in seconds):
    timechunks = dict()
    n = 0
    t_max = max(df.timestamp)
    t = min(df.timestamp)
    t_upper = t + dt
    num_chunks = (max(df.timestamp)-min(df.timestamp))/(dt)
    progress_old = 0
    progress_new = progress_old
    print("Chunking started at ", datetime.now())
    while t_upper < t_max:
        chunk = df[df.timestamp >= t]
        chunk = chunk[chunk.timestamp < t_upper]
        timechunks[n] = chunk
        progress_old = progress_new
        progress_new = round(100*n/num_chunks)
        if progress_new >= progress_old+1:
            print("Progress:", progress_new, '%', n, "out of approx.", int(num_chunks))
        t += dt
        t_upper = t + dt
        n += 1
    print("Chunking ended at ", datetime.now())
    return timechunks


timechunks = get_timelump(edgecsv, 300)
outfile = open('btdata_timechunks.pkl',"wb")
pickle.dump(timechunks, outfile)

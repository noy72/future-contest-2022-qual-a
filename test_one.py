from subprocess import Popen, PIPE, STDOUT
from multiprocessing import Pool
import pathlib
import sys

import pandas as pd

from tqdm import tqdm

run_test = ['cargo', 'run', '--release', '--bin', 'tester', './a.out']

def get_score(in_txt):
    p = Popen(run_test, stdout=PIPE, stdin=PIPE, stderr=PIPE, cwd='tools')
    result = p.communicate(input=in_txt)[1].decode()
    score = int(result.split('Score =')[1].strip())

    return score

def iterator(in_txt):
    for i in range(100):
        yield in_txt

if __name__ == '__main__':
    test_idx = sys.argv[1]
    
    scores = []
    with open(f'tools/in/{test_idx}.txt', 'br') as f:
        in_txt = f.read()

    with Pool(4) as p:
        for s in tqdm(
                p.imap_unordered(get_score, iterator(in_txt)),
                total=100
                ):
            scores.append(s)

    print(pd.Series(scores).describe())

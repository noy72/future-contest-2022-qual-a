from subprocess import Popen, PIPE, DEVNULL
from multiprocessing import Pool
import pathlib

import pandas as pd

from tqdm import tqdm

run_test = ['cargo', 'run', '--release', '--bin', 'tester', './a.out']

def get_score(txt):
    with open(txt, 'br') as f:
        in_txt = f.read()

    p = Popen(run_test, stdout=DEVNULL, stdin=PIPE, stderr=PIPE, cwd='tools')
    result = p.communicate(input=in_txt)[1].decode()
    score = int(result.split('Score =')[1].strip())
    if score == 0:
        print('\nError: ' + txt + '\n')

    return score

if __name__ == '__main__':
    scores = []
    input_file_pathes = [i for i in pathlib.Path('tools/in').glob('*.txt')]
    with Pool(4) as p:
        for s in tqdm(
                p.imap_unordered(get_score, input_file_pathes),
                total=len(input_file_pathes)
                ):
            scores.append(s)

    print(pd.Series(scores).describe())

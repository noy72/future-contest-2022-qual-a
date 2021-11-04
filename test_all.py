from subprocess import Popen, PIPE, STDOUT
from multiprocessing import Pool
import pathlib

from tqdm import tqdm

run_test = ['cargo', 'run', '--release', '--bin', 'tester', './a.out']

def get_score(txt):
    with open(txt, 'br') as f:
        in_txt = f.read()

    p = Popen(run_test, stdout=PIPE, stdin=PIPE, stderr=PIPE, cwd='tools')
    result = p.communicate(input=in_txt)[1].decode()
    score = int(result.split('Score =')[1].strip())

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

    print(f"""Result:
    max:    {max(scores)}
    min:    {min(scores)}
    mid:    {scores[len(scores) // 2]}
    ave:    {sum(scores) / len(scores)}""")

"""
生成した入力ファイルの値の記述統計を取る
"""

import sys

import pandas as pd


n, m, d, r = list(map(int, input().split()))
task_difficulty = []
for i in range(n):
    task_difficulty.append(list(map(int, input().split())))

task_dependency = [[] for _ in range(n)]
for i in range(r):
    temp = list(map(int, input().split()))
    task_dependency[temp[1] - 1].append(temp[0] - 1)

skill = []
for i in range(m):
    skill.append(list(map(int, input().split())))

time = []
for i in range(n):
    time.append(list(map(int, input().split())))


def mean_count(array):
    a = []
    for diff in array:
        s = pd.Series(diff)
        #print(s.sort_values())
        d = s.describe()
        a.append(d['50%'])
        #print(d['mean'])

    print(pd.Series(a).value_counts().sort_index())
    print(pd.Series(a).value_counts().sort_index().cumsum())


def sum_vals(array):
    a = []
    for diff in array:
        s = pd.Series(diff)
        a.append(s.sum())
        #print(d['mean'])

    print(pd.Series(a).value_counts().sort_index())

def std_vals(array):
    a = []
    for diff in array:
        s = pd.Series(diff)
        a.append(round(s.std(), 0))

    print(pd.Series(a).value_counts().sort_index())


def time_mean():
    t = [[] for i in range(m)]  # 転置
    for i, r in enumerate(time):
        for j, v in enumerate(r):
            t[j].append(v)

    index = []
    value = []
    for idx, v in enumerate(t):
        index.append(sum(skill[idx]))
        value.append(pd.Series(v).mean())

    print(pd.Series(value, index=index).sort_index())

time_mean()

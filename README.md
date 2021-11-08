# HACK TO THE FUTURE 2022 予選

## コード片
### 削減度を出力
```c++
  for (auto worker : workers) {
    vector<double> res;
    show("hoge") for (auto p : worker.finished_tasks) {
      int idx, a_days;
      tie(idx, a_days) = p;

      auto level = tasks[idx].level;
      double sum = accumulate(all(level), 0.0);
      res.emplace_back(a_days / sum);
    }
    sort(all(res));
    show(res);
  }
```

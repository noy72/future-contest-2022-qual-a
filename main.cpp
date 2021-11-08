#include <assert.h>
#include <stdio.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define range(i, a, b) for (int i = (a); i < (b); i++)
#define rep(i, b) for (int i = 0; i < (b); i++)
#define ranger(i, a, b) for (int i = (a)-1; i >= (b); i--)
#define repr(i, b) for (int i = (b)-1; i >= 0; i--)
#define all(a) (a).begin(), (a).end()
#define show(x) cerr << #x << " = " << (x) << endl;
using namespace std;

template <typename X, typename T>
auto vectors(X x, T a) {
  return vector<T>(x, a);
}
template <typename X, typename Y, typename Z, typename... Zs>
auto vectors(X x, Y y, Z z, Zs... zs) {
  auto cont = vectors(y, z, zs...);
  return vector<decltype(cont)>(x, cont);
}

template <typename T>
ostream& operator<<(ostream& os, vector<T>& v) {
  rep(i, v.size()) { os << v[i] << (i == v.size() - 1 ? "" : " "); }
  return os;
}
template <typename T, typename S>
ostream& operator<<(ostream& os, pair<T, S>& p) {
  os << '(' << p.first << ',' << p.second << ')';
  return os;
}
template <typename T>
istream& operator>>(istream& is, vector<T>& v) {
  for (T& x : v) {
    is >> x;
  }
  return is;
}

random_device seed_gen;
default_random_engine engine(seed_gen());

int randint(int a, int b) {
  uniform_int_distribution<> dist(a, b);
  return dist(engine);
}

double randdouble(double a, double b) {
  uniform_real_distribution<> dist(a, b);
  return dist(engine);
}

int min_element(const vector<int>& v) {
  int mini = INT_MAX;
  for (auto x : v) {
    mini = min(mini, x);
  }
  return mini;
}

int max_element(const vector<int>& v) {
  int maxi = 0;
  for (auto x : v) {
    maxi = max(maxi, x);
  }
  return maxi;
}

// ルーレット選択をする
int roulette(vector<int>& v) {
  int total = accumulate(all(v), 0);
  int threshold = randint(0, total - 1);

  int sum = 0;
  rep(i, v.size()) {
    sum += v[i];
    if (sum >= threshold) {
      return i;
    }
  }

  throw runtime_error("ルーレット選択のパラメータが 0 のみを含む配列です");
}
int roulette(vector<double>& v) {
  double total = accumulate(all(v), 0.0);
  double threshold = randdouble(0, total - 1);

  int sum = 0;
  rep(i, v.size()) {
    sum += v[i];
    if (sum >= threshold - 1e8) {
      return i;
    }
  }

  throw runtime_error("ルーレット選択のパラメータが 0 のみを含む配列です");
}

// ランダムにスキルベクトルを生成する
vector<int> generate_skill(int k) {
  normal_distribution<> dist(0.0, 1.0);

  vector<double> dds(k);
  rep(i, k) dds[i] = abs(dist(engine));

  vector<double> pdds(k);
  rep(i, k) pdds[i] = dds[i] * dds[i];

  double a = randdouble(20, 60);
  double b = sqrt(accumulate(all(pdds), 0.0));
  double q = a / b;

  vector<int> s(k);
  rep(i, k) s[i] = q * dds[i];
  return s;
}

class PredictedSkill {
 public:
  vector<int> skill;

  PredictedSkill(int k) : skill(vector<int>(k, 0)) {}

  // 与えたタスクの予測完了日数を返す
  int predict_days(const vector<int>& level) {
    int res = 0;
    rep(i, level.size()) { res += max(0, level[i] - skill[i]); }
    return res == 0 ? 1 : res;
  }
};

int predict_days(const vector<int>& level, const vector<int>& skill) {
  int res = 0;
  rep(i, level.size()) { res += max(0, level[i] - skill[i]); }
  return res == 0 ? 1 : res;
}

class Task {
 public:
  vector<int> level;
  int total_level;
  vector<int> successor_tasks;  // このタスクの後継タスク
  int predecessors_count;
  bool assigned;
  bool finished;

  Task()
      : successor_tasks(vector<int>()),
        predecessors_count(0),
        assigned(false),
        finished(false) {}

  void assign() { assigned = true; }

  void finish() {
    finished = true;
    // successro_tasks が示すタスクの predecessors_count を 1 減らす。
  }

  void setLevel(vector<int> in_level) {
    level = in_level;
    total_level = accumulate(all(level), 0);
  }
};

class Worker {
 public:
  vector<int> skill;
  PredictedSkill pred_skill;
  vector<pair<int, int>>
      finished_tasks;  // タスク番号と完了までにかかった日数のペア
  int assigned_task;
  int start_date;

  Worker(int k) : pred_skill(PredictedSkill(k)), assigned_task(-1) {
    skill = vector<int>(k, 0);
  }

  void assign_task(int day, int task) {
    start_date = day;
    assigned_task = task;
  }

  void finish_task(int day) {
    int took = day - start_date + 1;
    finished_tasks.emplace_back(make_pair(assigned_task, took));
    assigned_task = -1;
  }

  bool assignable() { return assigned_task == -1; }

  // 今までに完了したタスクレベルと完了までにかかった日数を用いて
  // pred_skill を更新する。
  void updatePredictedSkill(vector<Task>& all_tasks) {
    int task_size = finished_tasks.size();
    int skill_size = skill.size();

    vector<vector<int>> finished_task_levels(task_size);
    vector<int> actual_days(task_size);
    rep(i, task_size) {
      int idx, actual_day;
      tie(idx, actual_day) = finished_tasks[i];
      finished_task_levels[i] = all_tasks[idx].level;
      actual_days[i] = actual_day;
    }

    // 現在の予測スキルを元に、完了済みのタスクの予測完了日数を計算し、
    // pred_days を更新する。
    vector<int> pred_days(task_size);
    auto update_pred_days = [&] {
      rep(i, task_size) {
        pred_days[i] = pred_skill.predict_days(finished_task_levels[i]);
      }
    };
    update_pred_days();

    auto& pskill = pred_skill.skill;

    // 現在の予測スキルを元に、タスクレベルとの差を計算し、
    // diffs を更新する。
    // 予測スキル > タスクレベルのとき、差は正になる。　
    vector<vector<int>> diffs(task_size);
    auto update_diffs = [&] {
      rep(i, task_size) {
        vector<int> diff(skill_size);
        rep(j, skill_size) { diff[j] = pskill[j] - finished_task_levels[i][j]; }
        diffs[i] = diff;
      }
    };
    update_diffs();

    vector<int> candidate_skill = pred_skill.skill;
    int min_total_days = 0;
    rep(i, task_size) {
      min_total_days += abs(pred_skill.predict_days(finished_task_levels[i]) -
                            actual_days[i]);
    }

    // 完了日数が 3 以下のタスクについて、
    // そのタスクレベル以上のスキルは持っているとする。
    vector<int> lower_limit(skill_size, 0);
    for (auto p : finished_tasks) {
      int idx, actual_day;
      tie(idx, actual_day) = p;

      const vector<int>& finished_task_level = all_tasks[idx].level;
      if (actual_day <= 3) {
        rep(i, skill_size) {
          // [-3, 3]の誤差があるので、1 少なく考える
          lower_limit[i] = max(lower_limit[i], finished_task_level[i] - 1);
        }
      }
    }

    const int LOOP = 100;
    rep(_, LOOP) {
      PredictedSkill predicted_skill = PredictedSkill(skill_size);
      vector<int> generated_skill = generate_skill(skill_size);

      // 下限よりも小さい場合は下限に合わせる
      rep(i, skill_size) {
        if (generated_skill[i] < lower_limit[i]) {
          generated_skill[i] = lower_limit[i];
        }
      }

      predicted_skill.skill = generated_skill;

      int total_days = 0;
      rep(i, task_size) {
        total_days +=
            abs(predicted_skill.predict_days(finished_task_levels[i]) -
                actual_days[i]);
      }

      if (min_total_days > total_days) {
        min_total_days = total_days;
        copy(all(predicted_skill.skill), candidate_skill.begin());
      }
    }

    rep(i, LOOP) {
      auto copy_candidate_skill = candidate_skill;
      int idx = randint(0, static_cast<int>(copy_candidate_skill.size()) - 1);

      int cur = copy_candidate_skill[idx];
      copy_candidate_skill[idx] = randint(max(0, cur - 10), cur + 10);

      int total_days = 0;
      int nxt_total_days = 0;
      rep(i, task_size) {
        total_days +=
            abs(predict_days(finished_task_levels[i], candidate_skill) -
                actual_days[i]);
        nxt_total_days +=
            abs(predict_days(finished_task_levels[i], copy_candidate_skill) -
                actual_days[i]);
      }

      if (total_days > nxt_total_days) {
        candidate_skill = copy_candidate_skill;
      }
    }

    pskill = candidate_skill;
  }
};

int choice_worker(const vector<Worker>& workers, const vector<int>& level) {
  vector<pair<int, int>> free;  // ワーカーの id, 予測完了日数
  int min_predict_days = INT_MAX;
  rep(i, workers.size()) {
    auto worker = workers[i];
    if (worker.assignable()) {
      const int p_days = worker.pred_skill.predict_days(level);
      free.emplace_back(make_pair(i, p_days));
      min_predict_days = min(min_predict_days, p_days);
    }
  }
  if (free.size() == 0) return -1;

  vector<int> free_worker_idx;
  for (auto p : free) {
    if (p.second == min_predict_days) {
      free_worker_idx.emplace_back(p.first);
    }
  }

  assert(free_worker_idx.size() > 0);

  return free_worker_idx.at(randint(0, free_worker_idx.size() - 1));
}

int next_task(const vector<Task>& tasks) {
  int task_idx = -1, successor_task_count = -1, total_level = INT_MAX;
  vector<int> task_idx_list;
  vector<double> p;
  rep(i, tasks.size()) {
    auto task = tasks[i];
    if (task.predecessors_count > 0 or task.finished or task.assigned) continue;

    int size = static_cast<int>(task.successor_tasks.size());
    task_idx_list.emplace_back(i);
    p.emplace_back(task.total_level / (size + 1));
  }

  if (p.size() == 0) return -1;
  return task_idx_list[roulette(p)];
}

void solve() {
  int n, m, k, r;
  cin >> n >> m >> k >> r;

  vector<Worker> workers(m, Worker(k));

  vector<Task> tasks(n, Task());
  rep(i, n) {
    vector<int> level(k);
    cin >> level;
    tasks[i].setLevel(level);
  }

  rep(i, r) {
    int u, v;
    cin >> u >> v;
    u--;
    v--;

    tasks[v].predecessors_count++;
    tasks[u].successor_tasks.emplace_back(v);
  }

  int day = 0;
  while (true) {
    vector<pair<int, int>> assign_list;
    while (true) {
      int task_idx = next_task(tasks);
      if (task_idx == -1) break;

      int worker_idx = choice_worker(workers, tasks[task_idx].level);
      // cerr << task_idx << ' ' << worker_idx << endl;
      if (worker_idx == -1) break;

      workers[worker_idx].assign_task(day, task_idx);
      tasks[task_idx].assign();
      assign_list.emplace_back(make_pair(worker_idx + 1, task_idx + 1));
    }

    cout << assign_list.size();
    for (auto p : assign_list) {
      cout << ' ' << p.first << ' ' << p.second;
    }
    cout << endl << flush;

    int finish;
    cin >> finish;
    if (finish == -1) {
      break;
    }

    rep(i, finish) {
      int finished_worker;
      cin >> finished_worker;
      finished_worker--;

      auto& worker = workers[finished_worker];
      int finished_task_idx = worker.assigned_task;
      auto& task = tasks[finished_task_idx];

      worker.finish_task(day);
      worker.updatePredictedSkill(tasks);

      // 予測スキルを出力する
      cout << "#s " << finished_worker + 1 << ' ';
      for (auto v : worker.pred_skill.skill) {
        cout << v << ' ';
      }
      cout << endl;

      task.finish();
      for (auto suc : task.successor_tasks) {
        tasks[suc].predecessors_count--;
      }
    }
    day++;
  }
  /*
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
    transform(all(res), res.begin(), [](double x) { return 1 - x; });
    transform(all(res), res.begin(), [](double x) { return x * x; });
    show(res);
  }*/
}

int main(int args, char* argv[]) {
  auto begin = chrono::system_clock::now();
  solve();
  auto end = chrono::system_clock::now();
  double time = static_cast<double>(
      chrono::duration_cast<chrono::microseconds>(end - begin).count() /
      1000.0);
  show(time / 1000)
}

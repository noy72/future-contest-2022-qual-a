#define NDEBUG

#include <stdio.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <climits>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define range(i, a, b) for (int i = (a); i < (b); ++i)
#define rep(i, b) for (int i = 0; i < (b); ++i)
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

constexpr int max_n = 1000;
constexpr int max_m = 20;
constexpr int max_k = 20;
constexpr int max_r = 3000;

int finished_task_count;

template <typename T, int N>
struct Vector {
  T data[N];
  int size = 0;

  Vector() {}
  Vector(T x) {
    rep(i, N) {
      data[i] = x;
      size = N;
    }
  }

  inline T operator[](int index) const {
    assert(index < size);
    return data[index];
  }

  inline T& operator[](int index) {
    assert(index < size);
    return data[index];
  }

  inline void push(T x) {
    assert(size <= N);
    data[size++] = x;
  }

  inline void reset() { size = 0; }
};

template <typename T1, typename T2>
struct Duo {
  T1 first;
  T2 second;
  Duo() {}
  Duo(T1 a, T2 b) : first(a), second(b) {}
  bool operator<(const Duo& duo) const {
    if (first < duo.first) return true;
    if (first > duo.first) return false;
    if (second < duo.second) return true;
    return false;
  }
  bool operator>(const Duo& duo) const {
    if (first > duo.first) return true;
    if (first < duo.first) return false;
    if (second > duo.second) return true;
    return false;
  }
};

template <typename T1, typename T2, typename T3>
struct Trio {
  T1 first;
  T2 second;
  T3 third;
  Trio() {}
  Trio(T1 a, T2 b, T3 c) : first(a), second(b), third(c) {}
  bool operator<(const Trio& trio) const {
    if (first < trio.first) return true;
    if (first > trio.first) return false;
    if (second < trio.second) return true;
    if (second > trio.second) return false;
    if (third < trio.third) return true;
    return false;
  }
  bool operator>(const Trio& trio) const {
    if (first > trio.first) return true;
    if (first < trio.first) return false;
    if (second > trio.second) return true;
    if (second < trio.second) return false;
    if (third > trio.third) return true;
    return false;
  }
};

random_device seed_gen;
default_random_engine engine(seed_gen());

inline int randint(int a, int b) {
  uniform_int_distribution<> dist(a, b);
  return dist(engine);
}

inline double randdouble(double a, double b) {
  uniform_real_distribution<> dist(a, b);
  return dist(engine);
}

// 拡張 for はなんとなく使わない
int min_element(const vector<int>& v) {
  int mini = INT_MAX, n = v.size();
  rep(i, n) { mini = min(mini, v[i]); }
  return mini;
}

int max_element(const vector<int>& v) {
  int maxi = 0, n = v.size();
  rep(i, n) { maxi = max(maxi, v[i]); }
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

// タスクレベルが level, ワーカーのスキルが skill であるとき、
// タスクを完了するまでにかかる日数を返す。
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
  vector<int> pred_skill;
  vector<pair<int, int>>
      finished_tasks;  // タスク番号と完了までにかかった日数のペア
  int assigned_task;
  int start_date;

  Worker(int k) : pred_skill(vector<int>(k, 0)), assigned_task(-1) {
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
        pred_days[i] = predict_days(finished_task_levels[i], pred_skill);
      }
    };
    update_pred_days();

    /*
    // 現在の予測スキルを元に、タスクレベルとの差を計算し、
    // diffs を更新する。
    // 予測スキル > タスクレベルのとき、差は正になる。　
    vector<vector<int>> diffs(task_size);
    auto update_diffs = [&] {
      rep(i, task_size) {
        vector<int> diff(skill_size);
        rep(j, skill_size) { diff[j] = pred_skill[j] -
    finished_task_levels[i][j]; } diffs[i] = diff;
      }
    };
    update_diffs();
    */

    // 予測完了日数の変化がないため
    // 完了済みのタスクレベルよりも大きな変化は起こさない。
    Vector<int, max_k> limit(0);
    rep(i, task_size) {
      rep(j, skill_size) {
        limit[j] = max(limit[j], finished_task_levels[i][j]);
      }
    }

    int min_total_days = 0;
    rep(i, task_size) {
      min_total_days += abs(predict_days(finished_task_levels[i], pred_skill) -
                            actual_days[i]);
    }
    vector<int> candidate_skill = pred_skill;

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

    int LOOP = 100;
    rep(_, LOOP) {
      vector<int> generated_skill = generate_skill(skill_size);

      // 下限よりも小さい場合は下限に合わせる
      rep(i, skill_size) {
        if (generated_skill[i] < lower_limit[i]) {
          generated_skill[i] = lower_limit[i];
        }
      }

      // 上限よりも大きい場合は上限に合わせる
      rep(i, skill_size) {
        if (generated_skill[i] > limit[i]) {
          generated_skill[i] = limit[i];
        }
      }

      int total_days = 0;
      rep(i, task_size) {
        total_days +=
            abs(predict_days(finished_task_levels[i], generated_skill) -
                actual_days[i]);
      }

      if (min_total_days > total_days) {
        min_total_days = total_days;
        candidate_skill = generated_skill;
      }
    }

    rep(_, LOOP) {
      auto generated_skill = candidate_skill;

      // レベルにスキルを近づけるときは実数で計算したいので型を変える
      vector<double> d_generated_skill(skill_size);
      rep(j, skill_size) d_generated_skill[j] = generated_skill[j];
      rep(i, task_size) {
        // i 番目に完了したタスクのレベルの総和
        double total_level = all_tasks[finished_tasks[i].first].total_level;
        vector<int> tmp(skill_size);  // int じゃないと predict_days に渡せない
        rep(j, skill_size) tmp[j] = d_generated_skill[j];
        double predicted_days = predict_days(finished_task_levels[i], tmp);

        // この値が大きいほど、その i
        // 番目のタスクのレベルよりもスキルの方が大きい
        double projected_reduction_rate = (1 - predicted_days / total_level);
        projected_reduction_rate = projected_reduction_rate *
                                   projected_reduction_rate *
                                   projected_reduction_rate;

        // TODO: projected_reduction_rate が小さい場合は、
        // スキルが高すぎるかもしれないので、スキルを下げる

        rep(j, skill_size) {
          double diff = finished_task_levels[i][j] - d_generated_skill[j];
          if (diff <= 0)
            continue;  // レベルよりスキルの方が高いので、近づけようがない

          double amount = randdouble(0, diff) * projected_reduction_rate;
          d_generated_skill[j] += amount;
        }
      }

      vector<int> tmp(skill_size);
      int total_days_ = 0;
      rep(i, task_size) {
        total_days_ +=
            abs(predict_days(finished_task_levels[i], generated_skill) -
                actual_days[i]);
      }
      rep(j, skill_size) tmp[j] = d_generated_skill[j];
      int total_days = 0;
      rep(i, task_size) {
        total_days +=
            abs(predict_days(finished_task_levels[i], tmp) - actual_days[i]);
      }

      if (total_days_ > total_days) {
        generated_skill = tmp;
      }

      if (min_total_days > total_days) {
        min_total_days = total_days;
        candidate_skill = generated_skill;
      }
    }

    int candidate_skill_days_diff = 0;
    rep(i, task_size) {
      candidate_skill_days_diff +=
          abs(predict_days(finished_task_levels[i], candidate_skill) -
              actual_days[i]);
    }

    rep(_, LOOP * 20) {
      int idx = randint(0, skill_size - 1);

      // TODO: cur いらないかも。単に 0 ~ limit[idx] でランダムでいい？
      int cur = candidate_skill[idx];
      int changed_mono_skill = randint(0, limit[idx] + 1);

      int changed_skill_day_diff = 0;
      int candidate_skill_day_diff = 0;
      Vector<int, max_n> next_pred_days;
      rep(i, task_size) {
        int mono_level = finished_task_levels[i][idx];
        int mono_act_days = actual_days[i];
        int mono_pred_days = pred_days[i];

        int x = max(0, mono_level - candidate_skill[idx]);
        int y = max(0, mono_level - changed_mono_skill);

        int next_mono_pred_days = mono_pred_days - x + y;
        next_pred_days.push(next_mono_pred_days);
        changed_skill_day_diff += abs(mono_act_days - next_mono_pred_days);

        candidate_skill_day_diff += abs(mono_act_days - mono_pred_days);
      }

      if (candidate_skill_day_diff > changed_skill_day_diff) {
        candidate_skill[idx] = changed_mono_skill;
        rep(i, task_size) { pred_days[i] = next_pred_days[i]; }
      }
    }

    pred_skill = candidate_skill;
  }
};

int choice_worker(const vector<Worker>& workers, const vector<int>& level) {
  vector<pair<int, int>> free;  // ワーカーの id, 予測完了日数
  int min_predict_days = INT_MAX;
  rep(i, workers.size()) {
    auto worker = workers[i];
    if (worker.assignable()) {
      const int p_days = predict_days(level, worker.pred_skill);
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
    p.emplace_back(size);
  }
  if (p.size() > 0) {
    return task_idx_list[roulette(p)];
  }

  p = vector<double>();
  rep(i, tasks.size()) {
    auto task = tasks[i];
    if (task.predecessors_count > 0 or task.finished or task.assigned or
        task.successor_tasks.size() > 0)
      continue;

    task_idx_list.emplace_back(i);
    p.emplace_back(task.total_level);
  }
  if (p.size() > 0) {
    return task_idx_list[roulette(p)];
  }

  return -1;
}

vector<int> task_weight(const vector<vector<int>>& edge) {
  vector<Duo<int, int>> w(max_n, Duo(0, 0));
  rep(i, max_n) w[i].second = i;
  rep(from, max_n) {
    if (edge[from].size() == 0) continue;
    w[from].first++;
    rep(i, edge[from].size()) {
      int to = edge[from][i];
      if (from < to) w[to].first = max(w[from].first, w[to].first);
    }
  }

  repr(to, max_n) {
    rep(i, edge[to].size()) {
      int from = edge[to][i];
      if (from < to) w[from].first = max(w[from].first, w[to].first - 1);
    }
  }

  rep(i, max_n) {
    if (w[i].first == 0) w[i].first = 100;
  }

  sort(all(w));
  rep(i, max_n) { cerr << w[i].first << ' '; }
  cerr << endl;
  vector<int> t(max_n);
  rep(i, max_n) t[i] = w[i].second;

  return t;
}

vector<int> new_next_task(const vector<Task>& tasks,
                          const vector<int>& priority) {
  vector<int> res;
  rep(i, priority.size()) {
    int task_idx = priority[i];
    if (not tasks[task_idx].assigned and not tasks[task_idx].finished and
        tasks[task_idx].predecessors_count == 0)
      res.emplace_back(task_idx);
  }
  return res;
}

vector<Duo<int, int>> new_choice_worker(vector<Worker>& workers,
                                        vector<Task>& tasks,
                                        const vector<int>& next_tasks) {
  vector<Duo<int, int>> res;
  rep(i, max_m) {
    if (not workers[i].assignable()) continue;
    int limit = min(10, static_cast<int>(next_tasks.size()));

    int mini = INT_MAX;
    int task_idx = -1;
    int l = limit;
    int cnt = 0;
    rep(j, next_tasks.size()) {
      if (tasks[next_tasks[j]].assigned) continue;
      cnt++;

      if (not l-- && task_idx != -1) break;

      int pred_days =
          predict_days(tasks[next_tasks[j]].level, workers[i].skill);
      if (mini >= pred_days) {
        mini = pred_days;
        task_idx = next_tasks[j];
      }
    }
    show(cnt);
    if (task_idx == -1) continue;

    res.emplace_back(Duo(i, task_idx));
    tasks[task_idx].assign();
  }

  return res;
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

  vector<vector<int>> edge(n);
  rep(i, r) {
    int u, v;
    cin >> u >> v;
    u--;
    v--;

    edge[u].emplace_back(v);
    edge[v].emplace_back(u);

    tasks[v].predecessors_count++;
    tasks[u].successor_tasks.emplace_back(v);
  }

  auto priority = task_weight(edge);

  finished_task_count = 0;

  int day = 0;
  // Vector<Duo<int, int>, max_m> assign_list;
  int task_idx, worker_idx, finish;
  int finished_worker, finished_task_idx;
  int weight_idx = 0;
  while (true) {
    // assign_list.reset();
    auto next_tasks = new_next_task(tasks, priority);
    if (next_tasks.empty()) {
      cout << 0 << endl;
    } else {
      // worker_idx = choice_worker(workers, tasks[task_idx].level);
      auto assign_list = new_choice_worker(workers, tasks, next_tasks);
      // cerr << task_idx << ' ' << worker_idx << endl;

      // assign_list.push(Duo(worker_idx + 1, task_idx + 1));
      for (auto d : assign_list) {
        workers[d.first].assign_task(day, d.second);
        tasks[d.second].assign();
      }

      cout << assign_list.size();
      rep(i, assign_list.size()) {
        cout << ' ' << assign_list[i].first + 1 << ' '
             << assign_list[i].second + 1;
      }
      cout << endl << flush;
    }

    cin >> finish;
    if (finish == -1) {
      break;
    }

    rep(i, finish) {
      cin >> finished_worker;
      finished_worker--;

      auto& worker = workers[finished_worker];
      finished_task_idx = worker.assigned_task;
      auto& task = tasks[finished_task_idx];

      worker.finish_task(day);
      worker.updatePredictedSkill(tasks);

      // 予測スキルを出力する
      cout << "#s " << finished_worker + 1 << ' ';
      rep(i, k) { cout << worker.pred_skill[i] << ' '; }
      cout << endl;

      task.finish();
      finished_task_count++;
      rep(i, task.successor_tasks.size()) {
        tasks[task.successor_tasks[i]].predecessors_count--;
      }
    }
    day++;
  }
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

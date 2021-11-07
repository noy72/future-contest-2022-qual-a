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

class PredictedSkill {
 public:
  vector<int> skill;

  PredictedSkill(int k) : skill(vector<int>(k, 0)) {}

  // 与えたタスクの予測完了日数を返す
  int predict_days(vector<int>& task) {
    int res = 0;
    rep(i, task.size()) { res += max(0, task[i] - skill[i]); }
    return res == 0 ? 1 : res;
  }
};

class Task {
 public:
  vector<int> level;
  int total_level;
  vector<int> successor_tasks;  // このタスクの後継タスク
  int predecessors_count;
  bool assigned;
  bool finished;

  Task(int n, int r) : predecessors_count(0), assigned(false), finished(false) {
    level = vector<int>(n, 0);
    successor_tasks = vector<int>();
  }

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

    const int LOOP = 10;
    rep(_, LOOP) {
      int match_count =
          0;  // 予測完了日数と実際の完了日数の差が 3 以内であるものの数
      rep(i, task_size) {
        const vector<int>& diff = diffs[i];
        const vector<int>& finished_task_level = finished_task_levels[i];
        const int p_days = pred_days[i];
        const int a_days = actual_days[i];

        const int diff_pred_actual = p_days - a_days;

        // 完了日は [-3, 3] のズレがあるので、
        // 差がこの値以内あれば一致しているとみなす
        if (abs(diff_pred_actual) <= 3) {
          match_count++;
          continue;
        }

        // 完了日数が想定よりも短かったとき
        // => 想定スキルが実際よりも低い場合
        if (diff_pred_actual > 0) {
          vector<int> for_roulette(skill_size);
          rep(j, skill_size) {
            for_roulette[j] = diff[j] < 0 ? abs(diff[j]) : 0;
          }

          int target_idx;
          try {
            target_idx = roulette(for_roulette);
          } catch (exception e) {
            show("予測スキルをあげる！");
            show(pred_days[i]) show(actual_days[i])
                show(finished_task_levels[i]) show(diffs[i])
                    show(pskill) throw exception();
          }
          pskill[target_idx]++;
        }

        // 完了日数が想定よりも長かったとき
        // => 想定スキルが実際よりも高い場合
        if (diff_pred_actual < 0) {
          vector<int> for_roulette(skill_size);
          int mini = min_element(diff);
          rep(j, skill_size) {
            // 差分が負になる個別スキルも減少させる対象なので
            // 下駄を履かせて 0 より大きい値のみを持つ数列する。
            // また、現在の予測スキルが 0 であったり、レベルが 0 の場合は
            // これ以上下げられないので対象にならないようにする。
            for_roulette[j] = (pskill[j] == 0 or finished_task_level[j] == 0)
                                  ? 0
                                  : diff[j] + abs(mini) + 1;
          }

          // 個別スキル >= 個別レベル を満たす個別スキルに関しては、
          // 値の大小によって確率的に選ぶ意味がないので、数値を揃える。
          int maxi = max_element(for_roulette);
          rep(j, skill_size) {
            if (diff[j] >= 0) for_roulette[j] = maxi;
          }

          int target_idx;
          try {
            target_idx = roulette(for_roulette);
          } catch (exception e) {
            show("予測スキルを下げる！");
            show(pred_days[i]) show(actual_days[i])
                show(finished_task_levels[i]) show(diffs[i]) show(pskill)
                    show(for_roulette) show(mini)

            {
              show(pred_skill.skill);
              int h = pred_skill.predict_days(finished_task_levels[i]);
              show(h)
            }

            throw exception();
          }

          // タスクレベルよりもスキルが高いとき、-1
          // してもコストが変わらない場合があるので、
          // コストを上げる場合はスキルを必ずタスクレベル以下にする。
          if (pskill[target_idx] >= finished_task_level[target_idx]) {
            pskill[target_idx] = finished_task_level[target_idx] - 1;
          } else {
            pskill[target_idx]--;
          }
        }
      }
      // pskill
      // が変更されたので、タスクレベルとの差分と予測完了日数との差分を更新する。
      update_diffs();
      update_pred_days();

      if (match_count == task_size) {
        break;
      }
    }
  }
};

int random_choice_worker(const vector<Worker>& workers) {
  vector<int> free;
  rep(i, workers.size()) {
    auto worker = workers[i];
    if (worker.assignable()) {
      free.emplace_back(i);
    }
  }

  if (free.size() == 0) return -1;

  return free[randint(0, free.size() - 1)];
}

int next_task(const vector<Task>& tasks) {
  int task_idx = -1, successor_task_count = -1, total_level = INT_MAX;
  rep(i, tasks.size()) {
    auto task = tasks[i];
    if (task.predecessors_count > 0 or task.finished or task.assigned) continue;

    int size = static_cast<int>(task.successor_tasks.size());

    if (total_level > task.total_level) {
      total_level = task.total_level;
      successor_task_count = size;
      task_idx = i;
    } else if (total_level == task.total_level and
               successor_task_count < size) {
      successor_task_count = size;
      task_idx = i;
    }
  }

  return task_idx;
}

void solve() {
  int n, m, k, r;
  cin >> n >> m >> k >> r;

  vector<Worker> workers(m, Worker(k));

  vector<Task> tasks(n, Task(n, r));
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
      int worker_idx = random_choice_worker(workers);
      // cerr << task_idx << ' ' << worker_idx << endl;
      if (task_idx == -1 or worker_idx == -1) break;

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
}

int main() {
  auto begin = chrono::system_clock::now();
  solve();
  auto end = chrono::system_clock::now();
  double time = static_cast<double>(
      chrono::duration_cast<chrono::microseconds>(end - begin).count() /
      1000.0);
  show(time / 1000)
}

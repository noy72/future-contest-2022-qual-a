#include<iostream>
#include<stdio.h>
#include<vector>
#include<random>

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
	for (T& x : v) { is >> x; }
	return is;
}

random_device seed_gen;
default_random_engine engine(seed_gen());

int randint(int a, int b) {
	uniform_int_distribution<> dist(a, b);
	return dist(engine);
}


class Worker {
	public:
		vector<int> skill;
		vector<double> predict;
		vector<pair<int, int>> finished_tasks;  // タスク番号、かかった日数
		int assigned_task;
		int start_date;

		Worker(int k) : assigned_task(-1) {
			skill = vector<int>(k, 0);
			predict = vector<double>(k, 0);
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

		bool assignable() {
			return assigned_task == -1;
		}
};

class Task {
	public:
		vector<int> level;
		vector<int> successor_tasks;  // このタスクが終わったら取り掛かれるタスク
		int predecessors_count;
		bool assigned;
		bool finished;

		Task(int n, int r) : predecessors_count(0), assigned(false), finished(false){
			level = vector<int>(n, 0);
			successor_tasks = vector<int>();
		}

		void assign() {
			assigned = true;
		}

		void finish() {
			finished = true;
			// successro_tasks の predecessors_count を 1 減らす
		}
};

int random_choice_worker(const vector<Worker> &workers) {
	vector<int> free;
	rep(i,workers.size()){
		auto worker = workers[i];
		if(worker.assignable()) {
			free.emplace_back(i);
		}
	}
	
	if(free.size() == 0) return -1;

	return free[randint(0, free.size() - 1)];
}

int next_task(const vector<Task> &tasks) {
	int task_idx = -1, successor_task_count = -1;
	rep(i,tasks.size()){
		auto task = tasks[i];
		if(task.predecessors_count > 0 or task.finished or task.assigned) continue;
		int size = static_cast<int>(task.successor_tasks.size());
		if(successor_task_count < size) {
			successor_task_count = size;
			task_idx = i;
		}
	}

	return task_idx;
}

void solve() {
	int n, m, k, r;
	cin >> n >>  m >> k >> r;

	vector<Worker> workers(m, Worker(k));

	vector<Task> tasks(n, Task(n, r));
	rep(i,n){
		vector<int> level(k);
		cin >> level;
		tasks[i].level = level;
	}

	rep(i,r){
		int u, v;
		cin >> u >> v;
		u--; v--;

		tasks[v].predecessors_count++;
		tasks[u].successor_tasks.emplace_back(v);
	}

	int day = 0;
	while(true){
		vector<pair<int, int>> assign_list;
		while(true) {
			int task_idx = next_task(tasks);
			int worker_idx = random_choice_worker(workers);
			// cerr << task_idx << ' ' << worker_idx << endl;
			if(task_idx == -1 or worker_idx == -1) break;

			workers[worker_idx].assign_task(day, task_idx);
			tasks[task_idx].assign();
			assign_list.emplace_back(make_pair(worker_idx + 1, task_idx + 1));
		}

		cout << assign_list.size();
		for(auto p : assign_list){
			cout << ' ' << p.first << ' ' << p.second;
		}
		cout << endl << flush;

		int finish;
		cin >> finish;
		if(finish == -1) {
			break;
		}

		rep(i,finish){
			int finished_worker;
			cin >> finished_worker;
			finished_worker--;

			auto &worker = workers[finished_worker];
			int finished_task_idx = worker.assigned_task;
			auto &task = tasks[finished_task_idx];

			worker.finish_task(day);
			task.finish();
			for(auto suc : task.successor_tasks) {
				tasks[suc].predecessors_count--;
			}
		}
	}
}

int main() {
	solve();
}

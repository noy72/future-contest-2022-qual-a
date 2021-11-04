.PHONY: co
co:
	g++ -std=c++1z main.cpp
	cp a.out tools/

.PHONY: test
test:
	(cd tools && cargo run --release --bin tester ./a.out < in/$(n).txt > out.txt)

.PHONY: n
n:
	cat main.cpp | pbcopy

.PHONY: nout
nout:
	cat ./tools/out.txt

.PHONY: test-all
test-all:
	python3 test_all.py

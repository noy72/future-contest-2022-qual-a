.PHONY: co
co:
	g++ -std=c++1z main.cpp
	rm -f tools/a.out
	cp a.out tools/

.PHONY: test
test:
	(cd tools && cargo run --release --bin tester ./a.out < in/$(n).txt > out.txt)

.PHONY: coa
coa:
	make co
	make test n=0000

.PHONY: n
n:
	cat main.cpp | pbcopy

.PHONY: nout
nout:
	cat ./tools/out.txt | pbcopy

.PHONY: test-all
test-all:
	python3 test_all.py

.PHONY: test-one
test-one:
	python3 test_one.py $(n)

.PHONY: note
note:
	jupyter lab

.PHONY: venv
venv:
	. venv/bin/activate

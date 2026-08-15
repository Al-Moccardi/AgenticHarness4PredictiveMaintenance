# A-RAD edge tier — convenience targets
PY ?= python3
MODEL ?= llama-3.2-3b-instruct
N ?= 8
SEED ?= 1

.PHONY: help install smoke sample detect interpret run stream stats clean

help:
	@echo "make install     install python deps"
	@echo "make smoke       run the 20-check guard suite (offline)"
	@echo "make sample N=8  print a stratified reference-unit sample"
	@echo "make detect      frozen detection over the whole test set"
	@echo "make run         detect + interpret + stats on a reference sample"
	@echo "                 (vars: N=<units> SEED=<n> MODEL=<ollama model>)"
	@echo "make stream      live collection-sim demo on the reference sample"
	@echo "make clean       remove results/"

install:
	$(PY) -m pip install -r requirements.txt

smoke:
	$(PY) -m arad_edge smoke

sample:
	$(PY) -m arad_edge sample --n $(N) --seed $(SEED)

detect:
	$(PY) -m arad_edge detect apply --out results/test_anomalies.csv

run:
	$(PY) scripts/run_reference.py --n $(N) --seed $(SEED) --model $(MODEL)

stream:
	$(PY) scripts/run_stream.py --n $(N) --seed $(SEED) --model $(MODEL)

stats:
	$(PY) -m arad_edge stats

clean:
	rm -rf results/* && touch results/.gitkeep

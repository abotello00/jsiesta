# jsiesta quality gates. `make check` is the CI entry point (docs/design/IMPLEMENTATION_PLAN.md §5.3, §5.4).
#
#   make venv      create .venv with pinned dependencies (uv)
#   make ffi       build cpp/build/libjsiesta_ffi (libxc from /opt/etsf or LIBXC_ROOT)
#   make lint      ruff
#   make typecheck mypy on jax_siesta/ and kg/
#   make docs      doxygen + sphinx -W (warnings are errors)
#   make tutorials convert docs/tutorials/*.py to notebooks and execute them (needs `make ffi`; ~12 minutes)
#   make kg        rebuild and atomically publish kg/current (also refreshes docs/architecture)
#   make test      full pytest suite, JUnit XML in reports/pytest.junit.xml
#   make test-refs same, but missing SIESTA references fail instead of skip
#   make kg-results re-ingest reports/pytest.junit.xml into the KG (pass/fail/skip per verified_by edge)
#   make refs-cache  run the SIESTA-parity tests that support replay and store each SIESTA run in refs_cache/
#   make check     lint + typecheck + docs + kg + test + kg-results

PY      ?= .venv/bin/python
PYTEST  ?= $(PY) -m pytest
REPORTS := reports
# Tests that get all their SIESTA data through the replayable runners (run_siesta_reference / run_siesta_directory).
REFS_CACHE_TESTS := tests/test_siesta_parity.py tests/test_na_pair_exact.py tests/test_dispersion.py::test_d3_matches_siesta \
    tests/test_mgga_soc.py::test_pr_mgga_si_bulk_noncol_and_soc_parity

.PHONY: venv ffi lint typecheck docs tutorials kg kg-results test test-refs refs-cache check clean

venv:
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -r requirements-dev.txt
	uv pip install --python $(PY) --no-deps -e .

ffi:
	cmake -S cpp -B cpp/build -DJSIESTA_USE_LIBXC=ON
	cmake --build cpp/build -j

lint:
	$(PY) -m ruff check .

typecheck:
	$(PY) -m mypy

docs:
	doxygen docs/Doxyfile > /dev/null
	$(PY) -m sphinx -W -q -b html docs docs/_build/html

# The tutorials are written as "percent" scripts (# %% cells). Each is converted to a notebook and executed, so the
# notebooks in the documentation always show real output. Commit the .py and the .ipynb together.
VENV_BIN := $(abspath $(dir $(PY)))
tutorials:
	cd docs/tutorials && for f in [0-9]*.py; do \
	    echo "== $$f"; \
	    $(VENV_BIN)/jupytext --quiet --to ipynb --set-kernel python3 $$f && \
	    PATH="$(VENV_BIN):$$PATH" $(VENV_BIN)/jupyter-execute --inplace --timeout=3600 $${f%.py}.ipynb || exit 1; \
	done

kg: docs
	$(PY) kg/build_kg.py

kg-results:
	$(PY) kg/build_kg.py --junit $(REPORTS)/pytest.junit.xml

test:
	mkdir -p $(REPORTS)
	$(PYTEST) -q -rfEs --durations=15 --junitxml=$(REPORTS)/pytest.junit.xml

test-refs:
	mkdir -p $(REPORTS)
	$(PYTEST) -q -rfEs --durations=15 --require-refs --junitxml=$(REPORTS)/pytest.junit.xml

# Fills refs_cache/ (JSIESTA_SIESTA_REF_CACHE) so the same tests can replay real SIESTA results on machines without a
# SIESTA binary (for example a GPU node). Only tests that obtain SIESTA data via run_siesta_reference or
# run_siesta_directory are cache-capable (REFS_CACHE_TESTS).
refs-cache:
	mkdir -p $(REPORTS) refs_cache
	JSIESTA_SIESTA_REF_CACHE=$(CURDIR)/refs_cache $(PYTEST) -q -rfEs --require-refs $(REFS_CACHE_TESTS) \
	    --junitxml=$(REPORTS)/pytest_refs_cache.junit.xml

check: lint typecheck kg test kg-results

clean:
	rm -rf docs/_build docs/_doxygen docs/_generated docs/tutorials/work .pytest_cache .mypy_cache .ruff_cache

# Parallel test execution plan

## Goal

Run Enge's pytest suite in multiple worker processes without changing test
meaning, native-backend coverage, generated listening artifacts, or the ability
to reproduce a failure serially. The current suite has CPU-heavy numerical tests,
so parallel execution should reduce feedback time on machines with spare cores.

The default developer command will be `uv run pytest`; pytest-xdist will select
workers. One-process reproduction remains `uv run pytest -n 0`.

## Current isolation assessment

| Resource | Current behavior | Parallel requirement |
| --- | --- | --- |
| Renderer, controls, snapshots, NumPy arrays, native buffers | Constructed per test | Keep all mutable state local to a test. |
| WAV regression files | Written below pytest's `tmp_path` | Continue using `tmp_path`, never a repository path. |
| Rust extension | Built by `uv sync`, then imported | Build/install before pytest starts; workers only import it. |
| NumPy/native matrix | Parameterized per test | Keep both backends mandatory in one suite. |
| FLAC demos | Atomically published under `.pytest_cache/d/audio` | Keep synth, LFO, and filter names distinct per backend. |
| pytest cache | Shared project cache | Never treat it as a test input or correctness oracle. |

The first implementation must not introduce a worker-global renderer, a shared
RNG, a shared temporary filename, or test order dependencies. Float64 audio must
continue to be compared before encoding.

## Implementation

1. Add `pytest-xdist` to Enge's `dev` dependency group only.
2. Add this pytest configuration to `pyproject.toml`:

   ```toml
   [tool.pytest.ini_options]
   testpaths = ["test"]
   addopts = "-n auto --dist=worksteal"
   ```

   `worksteal` balances the longer audio cases across workers. This matches
   uFor's established configuration.
3. Document `uv run pytest` and the serial reproduction command in the README.
   State that `uv sync` must finish before a run that needs a rebuilt extension.
4. First run with a fixed worker count:

   ```sh
   uv run pytest -n 2 --dist=worksteal
   ```

5. Compare serial and parallel collection, ensuring every NumPy/native case runs.
   Run the parallel full suite twice: the second run exercises a pre-existing
   pytest cache and catches publication or global-state races.
6. Measure serial, two-worker, and `auto` wall times on a quiet host. Record the
   host, worker count, test count, and elapsed time here. Retain the default only
   if it improves feedback without obscuring failures.

## Acceptance criteria

- `uv run pytest` passes with xdist and runs the complete collected suite.
- `uv run pytest -n 0` passes and remains the documented serial reproducer.
- Two consecutive parallel full-suite runs pass.
- The six NumPy/native synth, LFO, and filter demo FLACs are complete and
  losslessly verified after a parallel run.
- Native no-fallback tests still prove Rust performed DSP.
- Ruff, Ty, Rust formatting, Clippy, and `git diff --check` retain their current
  verification roles.

## Failure handling

For a parallel-only failure, rerun the exact node ID with `-n 0`, then `-n 2`.
Inspect paths, cache use, randomness, module globals, and native-buffer ownership.
Fix the isolation defect and add a regression. Do not mark the test serial, add
retries, or weaken numerical tolerances. Give any future artifact a stable name
derived from test identity and backend, retaining atomic publication.

## Deliberate limits

This parallelizes pytest processes, not one audio render. It makes no claim about
real-time DSP performance, thread safety of one renderer instance, or concurrent
access to an output device.

## Additional work beyond the prompt

None.

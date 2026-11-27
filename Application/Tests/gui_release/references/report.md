# TRex GUI release run

- Run ID / UTC start and finish:
- Requested scope (all cases or named subset):
- Overall result: NOT_RUN
- Executable path (including resolved symlink/bundle executable):
- Executable SHA-256:
- Reported version / build configuration:
- Build revision supplied by user (unknown if unavailable):
- Conda environment / prefix / package inventory artifact:
- OS / architecture / display scale / window size:
- Resource root / layout and default-settings hashes:
- Suite revision or file hashes (separate from binary revision):
- Exact launch command, working directory, and environment overrides:
- Process/session/window identity:
- Startup overrides and reason for each:
- Fixture originals, copied paths, SHA-256 values, and settings edits:
- Additional model / partial-PV / hole fixture provenance:
- App log path / final exit status:
- Evidence limitations:

## Product findings before release

List each reproduced product problem, its impact, and the result that would
verify a fix. For each one, include:

- Case ID, build, environment, launch command, input and settings.
- Ordered actions from launch to the failure, including relevant timing.
- Expected and observed results and whether the failure reproduced.
- Screenshot, log excerpt, crash stack if available, and exact artifacts.

## Results

Use PASS, FAIL, BLOCKED, or NOT_RUN. Retain every row, including omitted cases.
List subcase results explicitly when they differ. Overall PASS requires all
cases and subcases to pass; report failures and incomplete coverage separately.

| ID | Case | Status | Evidence / observed values / blocker |
| --- | --- | --- | --- |
| GUI-01 | Startup | NOT_RUN | |
| GUI-02 | Fresh conversion and tracking | NOT_RUN | |
| GUI-03 | Cancel reconversion twice | NOT_RUN | |
| GUI-04 | Start over | NOT_RUN | |
| GUI-05 | Stop and Continue | NOT_RUN | Record PV lengths and saved tracking ranges before and after Continue. |
| GUI-06 | Paths, saved state, and configuration | NOT_RUN | |
| GUI-07 | Property editing, undo/redo, validation | NOT_RUN | |
| GUI-08 | Reset between files and resize | NOT_RUN | |
| GUI-09 | Outline alignment | NOT_RUN | |
| GUI-10 | Hole-outline alignment | NOT_RUN | |
| GUI-11 | Error recovery | NOT_RUN | |
| GUI-12 | Quit | NOT_RUN | |
| GUI-13a | Automatic raw-video startup without `-load` | NOT_RUN | Record both launches, the PV hash, and the reopened frame. |
| GUI-13b | Automatic raw-video startup with `-load` | NOT_RUN | Record both launches, the PV hash, and the restored saved frame. |

## Action trace

For each case, record its starting state, exact actions and entered values,
dialog choices, relevant frame/object IDs, elapsed waits, observed results, and
screenshot/log paths. Preserve the first failed attempt alongside any retries.

## Artifacts and coverage gaps

List created PV/settings/results/export files and evidence paths. Explain every
blocked/unrun subcase and any difference from the prepared fixture recipe.
For settings Undo and Redo, record the window readbacks, any Save State
prompt, and the results of the GUI-07 compiled fallback. Identify which
behavior each result exercised. Record transient preview glitches separately
from failures in the saved data or completed tracking.
If compiled tests ran, include the conversion round trip's result and which
output layouts exercised suppression.
Record the final process state. Keep the artifacts for reruns and comparison.

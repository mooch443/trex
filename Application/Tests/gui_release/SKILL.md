---
name: trex-gui-release-tests
description: Run the TRex GUI release regression suite on request against a specific Conda-built executable prepared by the user. Use for requests to test the real TRex window before a release, exercising conversion, cancellation, settings, file paths, scene transitions, outlines, and error recovery with screenshots and a saved report.
---

# TRex GUI release tests

Execute this suite only when the user asks for a run. Creating or updating the
suite does not launch TRex. This is an agent-driven native-window suite, using
clicks and keyboard input through Computer Use. It is not a CTest target.

## Select the prepared build

1. Use the exact executable or app bundle supplied for this run. If it is
   missing, ask for its absolute path and the Conda environment. Use only `trex`
   or `trex-modules` unless the user explicitly authorizes another environment.
   Do not build, install, update, or select a different executable from `PATH`.
2. Resolve a bundle to its actual executable for launch and hashing. Record its
   SHA-256, the reported version/build mode, Conda environment name and prefix,
   platform/architecture, and the user's build revision if supplied. An unknown
   build revision stays unknown; the checkout's HEAD does not identify the binary.
3. Use the resources shipped with that build. Record their effective location
   and hash the loaded layout JSON files and `default.settings` if present.
   Do not copy current source layouts into the installation. For a Conda package,
   resources can resolve to `CONDA_PREFIX/usr/share/trex`; `-wd` alone does not
   isolate them. TRex can update `.trex_recent_files` there during normal use.
4. Inspect the prepared build's defaults for automatic conversion, training,
   exports, or quit actions before launch. Disable such actions for this run
   using supported startup settings, recording every override. Keep setting
   values under test out of startup overrides so reload/reset tests remain valid.

## Prepare one run

From the repository root, create a new directory
`Application/Tests/gui_release/runs/<UTC timestamp>-<binary hash prefix>/`.
`runs/` is already ignored by this repository. Never reuse another run's output.
Copy [report.md](references/report.md) to `report.md` in that directory and fill
in the build identity before starting. Save the suite files or their hashes too.

Prepare `inputs/`, `outputs/`, and `screenshots/` beneath the run directory:

- Copy `videos/8guppies_20s.mp4` and `videos/test.settings` to matching basenames
  `inputs/nested folder/first.mp4` and `first.settings`. These supply a small
  background-subtraction conversion without downloading a model.
- Make a second copy, `second.mp4` with `second.settings`, for switching videos.
  Set `individual_prefix = "second-video"` and `track_threshold = 12` in this
  second settings copy. Keep other sample settings unchanged. Record these edits.
- Hash original and copied fixtures. All settings files, conversion products,
  caches, results, and exports used by the tests must belong to this run. Check
  copied settings for absolute paths before loading them.
- Generate the baseline PV through GUI-02 so its source points at the copied
  raw input. Do not assume an unrelated PV has a usable source path.
- For GUI-05, use a user-supplied resumable PV with its matching source/settings,
  stop a conversion through the GUI, or create a short PV using
  `video_conversion_range = [0,80]`. Record which preparation was used. The
  short-range proxy tests Continue and the extension boundary; the Stop control
  and partial-file finalization still need their own observed result.
- For GUI-10, use a supplied PV with visible mask holes and known frame/object
  IDs. For model-specific checks, use a supplied local model and compatible clip.
  Record missing fixtures as blocked; do not download a substitute model.
  The user-supplied [fish-school fixture](references/fish-school-fixture.md)
  covers both checks. Limit its conversions to frames `[0,99]`, including reruns.

## Run the real window

Read [cases.md](references/cases.md) before starting. Run every case unless the
user requests a subset; retain omitted rows as `NOT_RUN` with a reason. A subset
is not a complete release-suite pass.

Launch the selected executable with its Conda environment from the repository
root. For example, use `conda run --no-capture-output -n trex` followed by the
quoted absolute executable path and the recorded startup arguments. Capture
stdout/stderr in the run directory and retain the process/session identity and
eventual exit status. Do not supply a source initially: GUI-01 starts at the menu.
Do not use `-nowindow` or run commands from a build directory.

Use `cua_repl` for native UI interactions, following its current documentation:

- Identify the suite-owned window using the launched executable and visible
  version/title. Do not attach to or close a pre-existing personal TRex session.
- Prefer current accessibility elements when available. For custom-drawn
  controls, use coordinates derived from a fresh screenshot. Reobserve after
  actions and resize; never replay fixed coordinates from a previous run.
- Perform the actual user interactions. Shell reads may verify output files or
  logs; they must not replace a click, edit, save, or scene transition being tested.
- Before replacing text, verify the complete old value is selected or cleared.
  After committing, defocus and read the effective value back from the UI before
  starting the dependent operation. Do not assume modifier shortcuts or End
  worked. Preserve and correct malformed agent input without counting it as an
  application failure.
- A native file picker may be a separate window outside the TRex capture. Inspect
  the available surfaces before diagnosing a missing dialog. If it cannot be
  controlled, mark that route blocked and continue independent direct-path checks.
- Wait for observable readiness, conversion progress, and analysis completion.
  Record elapsed times. Use a 120-second no-progress deadline and a 10-minute
  per-case deadline unless the fixture needs a documented longer bound. A slow
  environment or unavailable dependency is `BLOCKED`; a reproducible application
  hang after otherwise successful setup is `FAIL`.
- Capture the named checkpoints in each case and every failure. Persist native
  screenshot bytes using the available tool's supported file facilities. If
  saving is unavailable, retain the tool observation and identify that evidence
  limitation in the report; do not invent a screenshot path.
- Keep an action trace with values, filenames, frame indices, timestamps, and
  dialog choices. Preserve a first failure before retrying; a later pass does
  not erase it. Continue independent cases when the application remains usable.

When a prerequisite fails, mark dependent cases `BLOCKED` and name that
dependency. Do not repair production code or alter the selected installation
to make the suite pass. A repaired/rebuilt binary needs a separate run.

## Finish and report

Close only the suite-owned process through the GUI, observe its exit, and finish
the Computer Use completion summary. Preserve run artifacts. Inspect outputs
and logs, and finish the report with one status per case and subcase:

- `PASS`: all stated assertions were observed, with evidence.
- `FAIL`: an observed application behavior violated an assertion.
- `BLOCKED`: a prerequisite or environment prevented observation.
- `NOT_RUN`: deliberately omitted or not reached; include the reason.

Call the complete suite passed only when every case and subcase passed. Link the
report, summarize behavior/UX failures and coverage gaps, and identify the exact
binary tested. Include complete reproduction steps for each failure. Opening
issues or changing application code is separate work, not part of this suite.

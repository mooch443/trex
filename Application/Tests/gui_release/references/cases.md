# GUI regression cases

Run in order, preserving the same application session where specified. Labels
below identify the intended controls; read the selected build's actual window
and bundled layouts when locating them. A missing or broken control is evidence,
not permission to invoke its internal action directly.

`RUN` denotes the new run directory. Record every changed setting and its starting
value. A pass requires the visible outcome, not just successful input delivery.

## GUI-01 — Startup

Launch the selected build without a source. Wait for the starting menu, inspect
its title/version, logo, text, and buttons, then open the configuration screen.

Assert: the intended binary opens a responsive native window; the main controls
are readable and usable; resources load without a blocking error. Capture
`01-startup` and the startup log/version. Record the window size/display scale.

## GUI-02 — Fresh conversion and tracking

1. Select the copied `first.mp4` through the input picker. Load the copied sample
   settings if they were not loaded automatically. Verify source, background
   subtraction, threshold, and tracking settings in the UI.
2. Leave the output folder/prefix at their default empty values. Record the
   resolved target before starting. If a default from the prepared installation
   points outside `RUN`, change it to the copied input folder and record that the
   default-location subcheck is blocked.
3. Click **Convert**. Observe progress and wait for completion. Open the resulting
   PV for tracking through the UI if the build does not transition automatically.
4. Wait for analysis, play and pause, and scrub near the start, middle, and end.

Assert: conversion produces a readable PV in the displayed target, tracking shows
moving objects, the timeline changes, and the last source frame is reachable.
Record frame count and output path. Capture `02-conversion` and `02-tracking`.
Keep this PV as the baseline for subsequent cases.

## GUI-03 — Cancel reconversion twice

1. Close the tracked file to the menu, then paste the baseline PV's absolute
   path in configuration. Record source, detector/model, output folder/prefix,
   `track_threshold`, and `individual_prefix`. Hash the PV before the attempt.
2. Click **Convert**. Verify the overwrite dialog names this PV, then choose
   **Cancel**. Inspect the same settings and selected file again.
3. Repeat Convert → Cancel in the same session. Then choose **Track** on the PV.

Assert: each cancel returns to a usable configuration with the recorded values;
it does not substitute the raw source for the selected PV, change the target, or
modify the PV. Tracking still works. Capture `03-before`, `03-dialog`, and
`03-after`. If a local model was supplied, repeat with its detector configuration
and record that subcase separately; otherwise report model preservation blocked.

## GUI-04 — Start over

Use a disposable copy of the baseline PV in its own folder under `RUN`; ensure
the dialog targets that copy. Change `video_conversion_range` to `[0,80]` through
the configuration UI. Click **Convert** → **Start over (overwrite)**, wait for
completion, and open the result for tracking.

Assert: only the displayed target was replaced; it opens and contains the chosen
range (81 frames when the source is long enough). The baseline remains unchanged.
Capture `04-overwrite-dialog` and `04-result`; record hashes and frame count.

## GUI-05 — Continue a partial conversion

1. Use a separate raw-input copy and output target. Start a full-range conversion,
   with `video_conversion_range = [-1,-1]`, then use its stop control and
   **Yes, stop** before completion. Record progress
   and wait for the saved partial PV. Alternatively use a supplied matching
   resumable fixture, recording its provenance and copied source path. If the
   sample finishes too quickly, create a short PV through the GUI with
   `video_conversion_range = [0,80]`, or reuse a copy of GUI-04's result. Preserve
   the short PV before extension and identify this as the range proxy.
2. Open the partial PV and record its last readable frame. Reopen configuration,
   request the same full range, and choose **Convert** → **Continue (if possible)**.
3. Wait for completion, track the result, and scrub across the recorded resume
   boundary and to the end.

Assert: the same output extends beyond its previous end; the prefix remains
readable, frames progress across the boundary without an obvious jump/repetition,
and the source end is reachable. Capture `05-partial`, `05-continue-dialog`, and
`05-complete`. Report Continue separately from Stop: the range proxy can pass the
extension assertions, while Stop and interrupted-file finalization remain
`BLOCKED` if they were not observed. A PV already containing the full requested
range cannot test extension. Do not use a forced process kill.

## GUI-06 — Paths, saved state, and configuration

1. Use another raw-input copy beneath a folder containing a space. Set an explicit
   output folder beneath `RUN/outputs` and `output_prefix = "prefixed"` in the UI.
   Convert and track. Record the resolved PV path; the prefix is a subdirectory.
2. Use **Menu → Save config** and **Save state**. Inspect the actual `.settings`
   and `.results` locations, then use **Load state** to reopen the saved state.
3. Close the file, paste the PV's absolute path, and track again. Exercise the
   **Results exist → Load Results** choice if offered. Repeat using the file picker.
4. Open the saved configuration through the settings-file picker and verify a
   recorded nondefault scalar and string value in the UI.

Assert: displayed and actual output paths agree, the explicit folder/prefix are
used, saved results reopen with the same data, and saved nondefault settings
reload. Neither opening route silently selects another PV with the same basename.
Capture `06-locations` and `06-reloaded`; list files created. Check saved keys or
effective UI values; PV metadata and `.settings` files are sparse deltas, not
complete configuration snapshots.

## GUI-07 — Property editing, undo, redo, and invalid input

In the tracking settings editor, record and change these values one at a time:

| Kind | Setting | Exercise |
| --- | --- | --- |
| Number | `track_threshold` | Change 12 to 13, or use another distinct valid pair. |
| Boolean | `gui_show_outline` | Toggle and observe outlines disappear/reappear. |
| String | `individual_prefix` | Set `gui-suite` and inspect identity labels. |
| List | `heatmap_ids` | Set `[0,1]`, then `[]`; read the committed values back. |
| Nested list | `output_fields` | Set `[["X",["wcentroid"]],["SPEED",["wcentroid"]]]` and read it back. |

After committing each edit, move focus out of the text editor. Use the app's
system-modifier+Z undo and system-modifier+Shift+Z redo (Command on macOS), verifying
the setting's value at each step. This must exercise application history, not
the textfield's local undo buffer. Leave the valid edits applied for the reload
check below.

Enter `not-a-number` for the numeric setting and a malformed bracketed value for
the list. Commit and inspect validation. Correct the value and continue editing.
Save and reload the valid string/list/nested-list changes using GUI-06's controls.
After verifying reload, restore each starting value through the UI.

Assert: committed values retain their type and contents; undo/redo restores the
expected values and visible boolean effect; invalid input is rejected without a
crash or unintended value change; valid input works afterward. Capture
`07-edits`, `07-undo-redo`, and `07-validation`. Record values after reopening.

## GUI-08 — Reset between files and preserve window usability

1. While tracking the first PV, change `individual_prefix` to `first-video` and
   `track_threshold` to 13 without saving those changes to the second fixture.
2. Pause playback and close menus/parameter popups. Capture `08-before-resize`
   and record the window dimensions, display scale, Parameters field width, and
   timeline height. Resize smaller (`08-smaller`), then restore the exact original
   dimensions on the same display (`08-restored-size`). Capture each result before
   moving the window or opening another file. The field width and timeline height
   must return to their original sizes, allowing only pixel rounding. Move the
   restored window without resizing it and capture `08-after-move`; control sizes
   must remain unchanged. A later recovery does not pass the restored-size check.
   Record `gui_interface_scale` at each checkpoint if exposed by the settings editor.
3. Use **Menu → Close file** to return to the starting screen. Open `second.mp4`
   and its prepared settings in configuration. Inspect `individual_prefix` and
   `track_threshold` before converting/tracking the second video.
4. Close and reopen the first PV, then navigate to the starting screen again.

Assert: the second file uses `second-video` and 12, the selected source changes
correctly, and the first file's edits do not leak into it. Menus, fonts, images,
and controls remain usable through both transitions without unexpected scale
changes at the same window size. Capture `08-before-close`, `08-second-settings`,
and `08-returned-menu`. Record any explicit carry-forward option selected.

## GUI-09 — Outline alignment

On a tracked copy of the sample, enable outline and posture display. Compare
`outline_approximate` values 0, 1, and 3 at the same recorded frame/object. If
the setting requires reanalysis, use the application's reanalysis controls before
comparing. Then play a short sequence and scrub forward/backward through it.

Assert: outlines remain centered on their objects without sudden translations,
and rendering remains finite and responsive. Smoothing may intentionally change
shape and detail. Capture `09-outline-0`, `09-outline-1`, and `09-outline-3` at
the same view, then restore the starting value. Do not infer alignment solely
from the absence of a crash.

## GUI-10 — Hole-outline alignment

Requires a copied PV with visible mask holes and recorded frame/object IDs.
Repeat GUI-09's comparisons with `outline_holes_approximate` values 0, 1, and 3.
Ensure the observations use newly calculated outlines, not cached old results.

Assert: holes remain associated with the same object and correctly positioned
within it, with no spikes or sudden translations during playback/scrubbing.
Capture `10-holes-0`, `10-holes-1`, and `10-holes-3`. A clip with no visible holes
cannot pass this case; report the missing prerequisite as `BLOCKED`.

## GUI-11 — Error recovery

1. In configuration, enter a nonexistent input path beneath `RUN`. Attempt the
   next operation if enabled. Observe the error/disabled control, then select
   the valid copied input and successfully enter conversion or tracking.
2. With a supplied local model and compatible clip, select the model-based
   detector and enter a nonexistent local `.pt` path beneath `RUN`. Inspect the
   prepared build's detector transition first: if it automatically loads a
   downloadable preset and no local route is available, mark this subcase blocked.
   Observe validation without choosing a downloadable preset.
   Correct it using a supplied compatible local model, or return to the known
   working background-subtraction configuration and continue.

Assert: the invalid source/model is explained or clearly rejected, does not
produce output at an unintended target, and the application remains usable.
The corrected operation succeeds in the same process. Capture `11-input-error`,
`11-model-error`, and `11-recovered`. Record a blocked model-validation subcase
if the prepared build lacks the required runtime; do not count switching back
to background subtraction as proof that a valid model loads.

## GUI-12 — Quit

From tracking, use **Menu → Quit**, choose **Cancel** once, and verify playback
or another control still works. Then quit and accept the exit dialog. Observe
the suite-owned process terminate and record the exit status and final logs.

Assert: cancelling quit preserves the session, confirmed quit closes it without
a hang, and a healthy run exits successfully. A previous application error must
remain in the report even if quit succeeds. Capture `12-quit-dialog`; record any
forced termination as a failure/blocker rather than a successful quit.

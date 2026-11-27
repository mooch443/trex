# GUI regression cases

Run in order, preserving the same application session where specified. Labels
below identify the intended controls; read the selected build's actual window
and bundled layouts when locating them. A missing or broken control is evidence,
not permission to invoke its internal action directly.

`RUN` denotes the new run directory. Record every changed setting and its starting
value. A pass requires the visible outcome, not just successful input delivery.

## GUI-01 — Startup

On macOS, record `pwd -P` in the launch shell outside the app bundle.
Record whether `imgui.ini` exists in the launch directory and, on macOS, in
`Contents/Resources`. Record the modification time of each existing file.

Launch the selected build without a source. Wait for the starting menu, inspect
its title/version, logo, text, and buttons, then open the configuration screen.

Once the macOS window is visible, verify that `lsof -a -p "$PID" -d cwd -Fn`
reports the recorded directory for the running TRex process.

Assert: the intended binary opens a responsive native window; the main controls
are readable and usable; resources load without a blocking error. Capture
`01-startup` and the startup log/version. Record the window size/display scale.

## GUI-02 — Fresh conversion and tracking

1. Paste the absolute path of the copied `first.mp4` into the source field and
   commit it. If the matching settings did not load automatically, enter their
   path in `settings_file` and reload the source. Verify source, background
   subtraction, threshold, and tracking settings in the UI.
2. Leave the output folder/prefix at their default empty values. Record the
   resolved target before starting. If a default from the prepared installation
   points outside `RUN`, change it to the copied input folder and record that the
   default-location subcheck is blocked.
3. Click **Convert**. Observe progress and wait for completion. Open the resulting
   PV for tracking through the UI if the build does not transition automatically.
4. Wait for analysis, play and pause, and scrub near the start, middle, and end.
5. On this unannotated recording, check the log for an unexpected
   `Failed to resolve FOI id for type annotated` error. In annotation mode,
   use **Add ∅ annotation** on one frame. Select **FOIs → annotated** and verify
   that frame is marked on the timeline. Use **Remove ∅ annotation** and verify
   the marker disappears.

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
2. Open the partial PV. Record its last readable frame and the range in its
   saved tracking state. Reopen configuration, request the same full range,
   and choose **Convert** → **Continue (if possible)**.
3. After completion, scrub from the end to the last frame of the partial PV
   and step to the first appended frame. Verify that both video frames are
   readable and that the identities and trails align on the newly tracked
   frames. Record the extended PV frame count and the first and last frames
   in the loaded `.results`. Continue currently supports a tracking state
   covering only the appended frames.
4. On a separate conversion, click **Stop** near the final frame and
   leave **End recording** open until the conversion finishes and tracking
   opens. Capture the open dialog with the current frame and total visible.
   If the guppy clip finishes too quickly, concatenate copies of it into a
   longer run-local video with the same settings. The confirmation should
   close automatically. If it remains visible,
   capture it and click **Yes, stop**. Verify tracking still responds, the
   completed PV contains its final frame, and a subsequent conversion starts
   and finishes normally. If the preceding recording's trails briefly appear
   while TRex computes the new background, capture them as a cosmetic
   observation. Check alignment once the new recording has tracked identities.
   Record the delayed confirmation separately from stopping an active conversion.

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
3. Close the file, paste the PV's absolute path into the source field, and track
   again. Exercise the **Results exist → Load Results** choice if offered.
4. In configuration, enter the saved settings path in `settings_file`, reload
   the source, and verify a recorded nondefault scalar and string value in the UI.
5. With an existing PV selected, choose SAM3 in Detection. Verify the prompt
   field appears and Locations names SAM3 in the summary. Restore the original mode.

Assert: displayed and actual output paths agree, the explicit folder/prefix are
used, saved results reopen with the same data, and saved nondefault settings
reload. The reopened PV must be the one at the entered path, even when another
PV has the same basename.
Capture `06-locations` and `06-reloaded`; list files created. Check saved keys or
effective UI values; PV metadata and `.settings` files are sparse deltas, not
complete configuration snapshots.

For a separate manual check, repeat the PV and settings reloads using the native
file choosers and verify the same recording and settings values.

## GUI-07 — Property editing, undo, redo, and invalid input

Pause on a tracked guppy frame with outlines visible. Show the timeline and
History panel. Record the starting values before changing the fish selection.
Commit one edit at a time.

| Kind | Setting | Exercise |
| --- | --- | --- |
| Number | `track_threshold` | Change 12 to 13, or use another distinct valid pair. |
| Boolean | `output_interpolate_positions` | Change the guppy fixture's `false` to `true`. |
| String | `individual_prefix` | Set `gui-suite` and inspect identity labels. |
| List | `categories_ordered` | Change `[]` to `["resting","swimming"]` and verify the order. |
| Nested list | `output_fields` | Set `[["X",["wcentroid"]],["SPEED",["wcentroid"]]]` and read it back. |

For each setting, verify the edit appears in TRex's History. Click an empty
part of the video to defocus the editor. Press Command+Z on macOS or Ctrl+Z
elsewhere, then read back the original value. Defocus again, add Shift to
the same shortcut, and read back the edited value. TRex logs `* reverted`
with the affected keys, or `* nothing to revert` for empty history.

If the first shortcut has no effect, use a disposable PV with an existing
`.results` file to check which key the tracking window receives as Z. With
the editor defocused, an unmodified Z opens **Overwrite** for Save State.
Cancel it. The active keyboard layout can map the driver's physical Y to Z.
Retry the modified shortcut with the confirmed key. If it also opens
**Overwrite**, cancel and stop retrying that chord. Record shortcut delivery
in the action trace and continue testing the remaining editors.

For that fallback, use the matching prepared test executables. Run `test_strings` with
`'--gtest_filter=GlobalSettingsTests.UnduOperation:GlobalSettingsTests.ComplexUnduOperation:GlobalSettingsTests.RemovesHistorySimple:GlobalSettingsTests.RemovesHistory:GlobalSettingsTests.DoesNotCrashWithEmptyHistory'`.
These tests call the same settings-history operations as the tracking scene
and assert restored and reapplied values. Run `test_gui_scroll` with
`'--gtest_filter=KeyboardEventTest.*'` to check layout and modifier translation.
Record the number of tests run and their results alongside the window edits. Credit an
end-to-end shortcut only after the window shows the original value on Undo
and the edited value on Redo. A handled shortcut that restores an incorrect
value is a product failure.

The tracking scene excludes `gui_*` and `heatmap_ids` from History. Toggle
`gui_show_outline` off and on and verify the fish outline disappears and
reappears. Set `heatmap_ids` to `[0,1]`, then `[]`, and read each value back.
Verify neither key appears in History after its edit.

Enter `not-a-number` for `track_threshold` and `[0,broken` for `heatmap_ids`.
Commit, defocus and reopen each editor. Its effective value must still be the
last valid value. Enter a valid value again and verify it commits.

Before saving, set `heatmap_ids` to `[0,1]` after the last fish selection.
Changing the selection synchronizes the list with the selected IDs. Save and
reload through GUI-06's controls. In Configuration, verify `individual_prefix`
is `gui-suite`, `heatmap_ids` is `[0,1]` and `output_fields` contains the two
edited entries before selecting another fish. Restore the starting
values through the UI. Capture `07-edits`, `07-undo-redo` and `07-validation`.

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
3. Press **R** to start screen recording. Wait for a `[rec] frame` line in the
   log, then choose **Menu → Close file** while recording. Verify that the log
   reports the clip was stopped and that the starting screen responds to input.
   Open `second.mp4`
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

Use a pose recording with a known frame where one individual's stored joint
visibility flags are all false and some joints have valid estimated positions.
Enable `gui_show_skeletons` and `gui_show_selections`, then select that individual.
Verify the selection outline encloses the blob. Valid hidden estimates should
appear as faint gray markers at their recorded positions. Any joint without a
valid position has no marker. Verify the individual has no identity-colored
joint markers or skeleton limbs. Record the input, frame and individual and
capture the selection.

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

After TRex exits, verify that the existence and modification time of each
`imgui.ini` path match the observations from GUI-01.

## GUI-13 — Automatic startup from a raw video

Run both subcases with separate copies of the suite MP4 and empty output folders.
In each subcase, relaunch with the same arguments and the same raw input path.

- **GUI-13a — Without `-load`.** Omit the flag on both launches.
- **GUI-13b — With `-load`.** Pass the flag on both launches.

1. Launch with `-i` pointing to the copied MP4, `-d` pointing to its output folder,
   `-detect_type background_subtraction`, `-video_conversion_range "[0,30]"`,
   and `-auto_quit false`. Leave `-task` unset and keep the native window open. Confirm
   the raw input in the startup log and verify conversion starts in the window
   without choosing a task. Wait for Tracking and verify that the expected PV
   exists with 31 frames.
2. After tracking finishes, pause playback, scrub to a nonzero frame, and record
   the displayed frame number. Use **Menu → Save state**. Verify the matching
   `.results` file exists and record the PV's SHA-256. Quit normally and retain
   the PV and saved results.
3. Relaunch from the same working directory. Verify the window opens directly
   in Tracking and the startup log identifies the expected PV. Verify its
   SHA-256 still matches. In GUI-13a, verify tracking starts fresh at frame 0
   and the log shows no successful read of the saved results. In GUI-13b, verify
   the saved frame is restored without a prompt and the log confirms a successful
   read of the matching `.results` file. Quit normally.

Capture the conversion and reopened Tracking windows for each subcase as
`13a-convert`, `13a-track`, `13b-convert`, and `13b-track`. Retain the launch logs
and record each subcase's result separately.

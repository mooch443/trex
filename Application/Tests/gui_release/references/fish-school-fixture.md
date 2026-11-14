# Local fish-school fixture

Use this fixture for the local semantic-model subcases of GUI-03 and GUI-11,
and for GUI-10 hole outlines. It supplements the small guppy fixture. Select
the executable and environment separately for each requested run.

## Inputs

- Video: `/Users/tristan/Downloads/sequence_20240303_060726300_DJI_0242.mp4`
- Settings: `/Users/tristan/Downloads/fish-schools-debug/sem/sequence_20240303_060726300_DJI_0242.settings`
- Local model referenced by those settings:
  `/Users/tristan/Downloads/school_segment.v13i.semantic-yolo26s-sem-1280px-1x-1787757756-semantic-d6a6e70b009a/weights/best.pt`

Recorded SHA-256 values (verify the supplied files on each run):

| Input | SHA-256 |
| --- | --- |
| Video | `6d5e29da987855d0e58c8bc3d88398a1357eeb294a8100ae581a187cca0985be` |
| Settings | `91b0a310d9f1f5b3f0426f5e727eb013718f8a7ca13f237a023c5fd0d5ed6ade` |
| Model | `8f8759e7fbb17cc1044aa34ed219fd2fb022b569dc334bebca832e0ba0a1e3e9` |

## Prepare and run

1. Copy the raw video, matching settings, and model into the new run directory.
   An independent APFS copy-on-write clone is suitable for the 13.2 GB video.
   Never write into the original video/settings/model directories.
2. In the settings copy, redirect `detect_model`, `meta_source_path`, and
   `output_dir` into the run. Preserve the other supplied values and annotations.
   Use `output_prefix = "sem"` for the baseline, with separate targets for other
   conversions. Record every edit and inspect any other absolute paths.
3. Select the copied raw input through the GUI. Set `video_conversion_range`
   to `[0,99]`, commit, defocus, and verify the effective range and output target
   before **every** conversion or reconversion. Recheck after any source change.
   The user requested a frame limit: never use the full 32,958-frame range for
   this fixture. For an explicitly requested command-line-source run, pass
   `-video_conversion_range '[0,99]'` at launch as well.
4. Verify that the local model loads and that the resulting PV has 100 frames.
   In the first verified run, the model was recognized as 1280 px / masks, and
   frame 75 showed two visible holes in school `id_0`. Find and record the actual
   frame/object on each build; do not assume that identity numbers are stable.
5. Exercise GUI-10 at that frame with `outline_holes_approximate` 0, 1, and 3,
   then playback/scrubbing and restoration. Use this same local model for the
   GUI-03 cancellation and GUI-11 invalid-model/recovery subcases.
6. If testing Continue with a range proxy, create `[0,49]` and extend only to
   `[0,99]`. Report actual Stop separately when the short conversion finishes
   before a stop can be observed. Do not increase the frame cap to slow it down.

Evidence for this recipe is retained in
`runs/20261113T131007Z-b6dfac85-schools/`: `manifest.json`, `checks.json`,
`actions.log`, `report.md`, and the GUI-03/10/11 screenshots. This local fixture
may be unavailable on other machines; record that prerequisite as blocked.

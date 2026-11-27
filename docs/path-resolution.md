# TRex recording and output paths

Keep this table and the path-resolution tests up to date together whenever the rules change.
Assume TRex was launched from `/work`. An em dash means the option was omitted.
If `launch_dir()` is empty, use the process's current working directory (`cwd()`) wherever these rules refer to the launch directory.
Relative inputs and relative output directories start at the launch directory.
The default output root is the original source's directory, or the launch directory for a camera.
`-p` appends a relative subdirectory to the output root. Outputs include settings, results and exports under `data/`.

| `-i` | `-p` | `-d` | Expected outcome |
|---|---|---|---|
| `clip.pv` | — | — | Open `/work/clip.pv`. Outputs in `/work/`. |
| `clip.pv` | `run` | — | Open `/work/clip.pv`. Outputs in `/work/run/`. |
| `rel/clip.pv` | `run` | — | Open `/work/rel/clip.pv`. Outputs in `/work/rel/run/`. |
| `/media/clip.pv` | `run` | — | Open `/media/clip.pv`. Outputs in `/media/run/`. |
| `/media/clip.pv` | — | `out` | Open `/media/clip.pv`. Outputs in `/work/out/`. |
| `/media/clip.pv` | `run` | `out` | Open `/media/clip.pv`. Outputs in `/work/out/run/`. |
| `rel/clip.pv` | `run` | `/exports` | Open `/work/rel/clip.pv`. Outputs in `/exports/run/`. |
| `clip.mov` | — | — | Use or create `/work/clip.pv`. Other outputs alongside it. |
| `clip.mov` | `run` | — | Use or create `/work/run/clip.pv`. Other outputs alongside it. |
| `rel/clip.mov` | `run` | — | Use or create `/work/rel/run/clip.pv`. Other outputs alongside it. |
| `/media/clip.mov` | `run` | — | Use or create `/media/run/clip.pv`. Other outputs alongside it. |
| `/media/clip.mov` | — | `out` | Use or create `/work/out/clip.pv`. Other outputs alongside it. |
| `/media/clip.mov` | `run` | `out` | Use or create `/work/out/run/clip.pv`. Other outputs alongside it. |
| `rel/clip.mov` | `run` | `/exports` | Use or create `/exports/run/clip.pv`. Other outputs alongside it. |
| `clip` | — | — | Look for the existing recording `/work/clip.pv`. |
| `clip` | `run` | — | Look only for `/work/run/clip.pv`. If it is missing, the recording shorthand fails. |
| `clip` | `run` | `/exports` | Look only for `/exports/run/clip.pv`. |
| `webcam` | `run` | `out` | The recording is `/work/out/run/webcam.pv`. `basler` follows the same rule. |
| — | `run` | `out` | Configure `/work/out/run/` for outputs. Tracking still requires a selected source. |

An explicit PV supplies both frames and PV metadata. A recording of the same name under the output directory cannot replace a missing explicit PV.
For a movie, AUTO tracks the existing derived PV or converts the raw source there if needed. Explicit tracking requires the recording to exist.
An output name (`-o`) is a literal basename for a raw source. Supplying it with a PV input is an error, as is explicitly requesting conversion of a PV.
A PV pattern must select one recording for tracking. A selection mixing raw inputs and PVs is invalid.

When the original movie has moved, the settings loader updates `meta_source_path` if it locates and opens the movie.
The animated background can also locate the movie for playback. It keeps that choice local and preserves both `source` and `meta_source_path`.

An absolute output prefix is invalid. Nested relative prefixes are supported.
With the same settings, passing a resolved path through the resolver again preserves the complete absolute path.
For example, `-i run/clip.mov -p run` gives `/work/run/run/clip.pv`. Each `run` has a distinct origin.
Resolving that PV or the export `/work/run/run/data/fish.csv` again preserves every component.

## What the callers request

| Call with the effective `sprite::Map` | Expected result |
|---|---|
| `parse(purpose, paths, &map)` with a `PathArray` | Another `PathArray`. Each matched file is passed as an absolute path to the scalar resolver for that purpose, with the same settings. |
| `parse("input", path, &map)` | Locates the supplied input. An explicitly selected PV provides the recording at that location. |
| `settings::choose_output_name(source)` | One default PV filename for the selected input files, including a multi-file movie sequence. |
| `settings::choose_output_name(map)` | Uses the explicit output name when supplied, otherwise derives the default from the selected inputs. |
| `parse("output", path, &map)` | Places the supplied name under the output root and prefix. A supplied absolute path keeps its location. The filename and extension are preserved. |

For `-i clip.mov`, conversion chooses the filename `clip.pv` before passing it to `parse("output", …)`. Passing `clip.mov` to that same location produces an output path ending in `clip.mov`.

For example, expanding `clips/*.mov` inside a `PathArray` yields relative names such as `clips/a.mov`.
When TRex launches from `/work`, `DataLocation::parse` passes `/work/clips/a.mov` to the registered resolver.
The returned array holds the resolver's results as literal paths. Camera tokens such as `webcam` are handled by TRex before file resolution.

## Testing one rule

Each table row has a named Google Test case for selecting the recording and another for placing outputs.
For example, after building the Debug tests, run either command from the project root.

```sh
Application/build/Debug/test_tracking_filename_resolution --gtest_filter=PathTable/RecordingPathRules.SelectsRecording/PvWithPrefix
Application/build/Debug/test_tracking_filename_resolution --gtest_filter=PathTable/RecordingPathRules.PlacesOutputs/PvWithPrefix
```

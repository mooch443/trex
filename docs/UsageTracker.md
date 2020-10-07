# Overview

This file contains some frequently asked questions and usage tips based
on my personal experience while developing and using the app.

I hope this will be of help to people of the future.

## Contents

1. [Keyboard shortcuts](#keys)
2. [Frequently asked questions](#faq)
3. [Typical use cases](#uses)
4. [Tips](#tips)

<a name="keys"></a>
# 1. Keyboard shortcuts
| Key       | Action                                                                                         |
|-----------|------------------------------------------------------------------------------------------------|
| Esc       | Terminate                                                                                      |
| Return    | Next frame                                                                                     |
| Backspace | Previous frame                                                                                 |
| Space     | Start/stop playback                                                                            |
| Comma     | Resume/pause analysis                                                                          |
| B         | Show/hide posture preview when an individual is selected                                       |
| T         | Show/hide timeline and some interface elements                                                 |
| G         | Show/hide graph when an individual is selected                                                 |
| D         | Switch to RAW mode and back                                                                    |
| M         | Jump to next frame where the number of recognized individuals changed (yellow in timeline)     |
| N         | Jump to previous frame where the number of recognized individuals changed (yellow in timeline) |
| R         | Playback frame-by-frame and save what you see in the tracker window as `<output_dir>/images`   |
| S         | Export data in `output_graphs` to CSV files in `<output_dir>/fishdata`                         |
| Z         | Save program state to `<videoname>.results`                                                    |
| L         | Load program state from `<videoname>.results`                                                  |
| I         | Save events to CSV files `<output_dir>/fishdata/<videoname>_events<fishid>.csv`                |
| P         | Cycle through fish based on ID

<a name="faq"></a>
# 2. FAQ
### How do I open a mp4/avi/mkv video?
The only way to open any video that has not been recorded using FrameGrabber, is to convert into the `.pv` format it first. 
This can be done using **FrameGrabber**. Some usage examples:

```
# convert an .mp4 file to .pv
./framegrabber -i /path/to/video.mp4 -o /output/path/video.pv -settings /path/to/conversion.settings

# convert a set of images00000.jpg to .pv
./framegrabber -i /path/to/images%05d.jpg -o /output/path/video.pv -settings /path/to/conversion.settings
```

The `conversion.settings` part is curcial here. A typical conversion settings file could look like this (also located in `Application/conversion.settings`):

```
frame_rate = 24
crop_offsets = [0.0, 0.0, 0.0, 0]
fish_minmax_size = [0.05, 100000]
threshold_constant = 30

meta_real_width = 50
cam_circle_mask = false
cam_undistort = false
recording = true
```

1. Setting the `frame_rate` is important for the tracking later on (e.g. in speed calculations)
2. `crop_offsets` will crop the video before converting it. Numbers are percentages [left,top,right,bottom].
3. `fish_minmax_size` sets the limits for what could be a fish and what is noise (For the conversion, this does not have to be accurate. Just use a very small and a big number, unless you're getting too much noise.)
4. The `threshold_constant` specifies the minimal difference between background and the fish color. Every pixel below that threshold will be discarded as background. Increase this number if you have too much background around your cropped out fish.
5. `meta_real_width` should be set the actual size of what is seen in the (cropped) video. So if the video is cropped to only the tank, this would be the width of the tank in cm.
6. Disable the `cam_circle_mask` unless your tank is round.
7. Disable `cam_undistort` unless you also set `cam_undistort1` and `cam_undistort1` as the two camera parameter matrices.
8. `recording` should be set to true in all conversion cases. This starts to immediately save to a file.

### What is the answer to the ultimate question of life, the universe, and everything?
Try searching the source code folder for words matching `DIEANTWOORD` and you will find your answer. If you are too lazy to do this yourself, you may also use this handy script snippet and execute it within the root folder of this repository:

```
for f in `find . -name '*.h'`; do 
    cat "$f" | grep DIEANTWOORD; 
done
```

<a name="uses"></a>
# 3. Typical use cases

<a name="tips"></a>
# 4. Tips

- The slowest part of the application is usually the posture analysis. So when analysing a file with **lots of individuals**, it is recommended to disable posture analysis using `calculate_posture = false`.
- When calculating **postures**, try looking at the posture preview on the top-right (enable it by clicking on an individual in the tracker). If the posture has lots of one color (red), but not of the other color (green), you should try changing `outline_curvature_range` until there are clear peaks of both colors.
- Other ways to **improve posture detection** is by smoothing the outline. If e.g. the posture is flipped quite often, then usually its because of lots of curvature peaks in the outline. Try setting `outline_smooth_samples` to a higher value.
- For really big fish on **high resolution videos**, it is recommended to increase `outline_resample` above `1`. For small fish, it is recommended to decrease it to a value between `0.1` and `1`.
- **Really big groups** of fish can be quite slow to track because of the many possible combinations of blobs and fish. Try increasing `matching_probability_threshold`.

@echo off
set PWD=%cd%\..\..\videos

tgrabs -d "%PWD%" -i "%PWD%/test_frames/frame_%%3d.jpg" -o test -threshold 9 -average_samples 100 -averaging_method mode -meta_real_width 30 -exec "%PWD%/test.settings" -enable_live_tracking -auto_no_results -auto_no_tracking_data false -nowindow -output_format csv -track_do_history_split false -track_threshold 0
if not errorlevel 0 (
    echo TGrabs execution failed.
    exit /b 1
)

echo Execution finished.

for %%f in (%PWD%/data/test_fish*.csv) do (
    echo Checking %%~nf
    git --no-pager diff --word-diff --no-index -- "%PWD%/data/%%~nf.csv" "%PWD%/compare_data/raw/%%~nf.csv"
    if not errorlevel 0 (
        echo Comparison failed for file '%PWD%/data/%%~nf.csv'
        exit /b 1
    )
)

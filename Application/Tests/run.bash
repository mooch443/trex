PWD=$(pwd)/../../videos
tgrabs -d "${PWD}" -i 8guppies_20s.mp4 -o test -threshold 9 -average_samples 100 -averaging_method mode -meta_real_width 30 -nowindow -s test -enable_live_tracking -auto_no_results
trex -d "${PWD}" -i test -s "${PWD}/test.settings" -p corrected -auto_apply -auto_quit -auto_no_results -nowindow

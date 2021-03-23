PWD=$(pwd)/../../videos

if ! tgrabs -d "${PWD}" -i 8guppies_20s.mp4 \
     -o test -threshold 9 -average_samples 100 -averaging_method mode -meta_real_width 30 -exec "${PWD}/test.settings" \
     -enable_live_tracking -auto_no_results -auto_no_tracking_data false -nowindow -output_format csv; then
    echo "TGrabs could not be executed."
    exit 1
fi

for f in `ls ${PWD}/data/test_fish*.csv`; do
    f=$(basename $f .csv)

    echo "Checking $f ..."
    if ! git --no-pager diff --no-index -- ${PWD}/data/${f}.csv ${PWD}/compare_data/raw/${f}.csv; then
        echo "files $f differ from baseline"
        exit 1
    fi
done

rm -rf ${PWD}/data

#trex -d "${PWD}" -i test -s "${PWD}/test.settings" -p corrected -auto_apply -auto_quit -auto_no_results -nowindow
#rm -rf ${PWD}/corrected/data
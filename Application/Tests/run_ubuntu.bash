PWD=$(pwd)/../../videos

if ! tgrabs -d "${PWD}" -i 8guppies_20s.mp4 \
     -o test -threshold 9 -average_samples 100 -averaging_method mode -meta_real_width 30 -exec "${PWD}/test.settings" \
     -enable_live_tracking -auto_no_results -auto_no_tracking_data false -nowindow; then
    echo "TGrabs could not be executed."
    exit 1
fi

for f in `ls ${PWD}/data/test_fish*.npz`; do
    f=$(basename $f .npz)

    unzip -d ${PWD}/data/$f -o ${PWD}/data/${f}.npz > /dev/null
    unzip -d ${PWD}/compare_data/raw/$f -o ${PWD}/compare_data/raw/${f}.npz > /dev/null

    MD5_CURRENT="$(md5sum -q ${PWD}/data/${f}/X.npy) $(md5sum -q ${PWD}/data/${f}/SPEED#wcentroid.npy)"
    MD5_COMPARE="$(md5sum -q ${PWD}/compare_data/raw/${f}/X.npy) $(md5sum -q ${PWD}/compare_data/raw/${f}/SPEED#wcentroid.npy)"

    echo "Checking '$f'... $MD5_COMPARE | $MD5_CURRENT"

    if [ "$MD5_COMPARE" != "$MD5_CURRENT" ]; then
        echo "md5sum for $f differs from baseline"
        exit 1
    fi

    rm -rf ${PWD}/compare_data/raw/${f}/
    rm -rf ${PWD}/data/${f}/
done

rm -rf ${PWD}/data

#trex -d "${PWD}" -i test -s "${PWD}/test.settings" -p corrected -auto_apply -auto_quit -auto_no_results -nowindow
#rm -rf ${PWD}/corrected/data
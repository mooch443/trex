#!/bin/bash

PWD=$(pwd)/../../videos
TGRABS=tgrabs
TREX=trex

#TGRABS=~/trex/Application/build/RelWithDebInfo/TGrabs.app/Contents/MacOS/TGrabs
#TREX=~/trex/Application/build/RelWithDebInfo/TRex.app/Contents/MacOS/TRex

CMD="${TGRABS} -d "${PWD}" -i \"${PWD}/test_frames/frame_%3d.jpg\" -o test -threshold 9 -average_samples 100 
    -averaging_method mode -meta_real_width 2304 -exec \"${PWD}/test.settings\" 
    -enable_live_tracking -auto_no_results -output_format csv -nowindow"
echo "Running TGrabs... ${CMD}"
if ! { ${CMD} 2>&1; } > "${PWD}/tgrabs.log"; then
    cat "${PWD}/tgrabs.log"
    echo "TGrabs could not be executed."
    exit_code=1
else
    echo "  Scanning files..."
    FILES=$(ls ${PWD}/data/test_fish*.csv)
    
    if [ -z "${FILES}" ]; then
        echo "[ERROR] No files found."
        exit_code=1
    else
        for f in ${FILES}; do
            f=$(basename $f .csv)

            echo -e -n "\tChecking $f ..."
            if ! git --no-pager diff --word-diff --no-index -- ${PWD}/data/${f}.csv ${PWD}/compare_data/raw/${f}.csv; then
                echo "FAIL"
                echo "[ERROR] file $f differs from baseline"
                exit_code=1
            else
                echo 'OK'
            fi
        done
    fi
fi

rm -rf ${PWD}/data
echo ""

MODES="automatic
hungarian
tree"

for MODE in ${MODES}; do
    CMD="${TREX} -d \"${PWD}\" -i test -s \"${PWD}/test.settings\" -p corrected -match_mode ${MODE} -auto_quit -auto_no_results -output_format csv -nowindow"

    echo "Running TRex (${MODE})... ${CMD}"

    if ! { ${CMD} 2>&1; } > "${PWD}/trex.log"; then
        cat "${PWD}/trex.log"
        echo "TRex could not be executed."
        exit_code=1
    else
        echo "  Scanning files..."
        FILES=$(ls ${PWD}/corrected/data/test_fish*.csv)
        
        if [ -z "${FILES}" ]; then
            echo "[ERROR] No files found."
            exit_code=1
        else
            for f in ${FILES}; do
                f=$(basename $f .csv)

                echo -e -n "\tChecking $f ..."
                if ! git --no-pager diff --word-diff --no-index -- ${PWD}/corrected/data/${f}.csv ${PWD}/compare_data/raw/${f}.csv; then
                    echo "FAIL"
                    echo "[ERROR] corrected file $f differs from baseline"
                    exit_code=1
                else
                    echo 'OK'
                fi
            done
        fi
    fi

    rm -rf ${PWD}/corrected/data
done

exit ${exit_code}

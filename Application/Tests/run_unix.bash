#!/bin/bash

PWD=$(pwd)/../../videos
WPWD=${PWD}
TGRABS=tgrabs
TREX=trex

echo "Detecting system..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Linux";
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "MacOS";
else 
    echo "Windows: ${WPWD}"
    if ! [ -f ./wslpath ]; then
        echo "Cannot find wslpath. Downloading..."
        WPWD=$(python wslpath -w ${WPWD})
    else
        WPWD=$(python wslpath -w ${WPWD})
    fi
fi

#TGRABS=~/trex/Application/build/RelWithDebInfo/TGrabs.app/Contents/MacOS/TGrabs
#TREX=~/trex/Application/build/RelWithDebInfo/TRex.app/Contents/MacOS/TRex

CMD="${TGRABS} -d "${WPWD}" -i \"${WPWD}/test_frames/frame_%3d.jpg\" -o test -threshold 9 -average_samples 100 
    -averaging_method mode -meta_real_width 2304 -exec \"${WPWD}/test.settings\" 
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
    CMD="${TREX} -d \"${WPWD}\" -i test -s \"${WPWD}/test.settings\" -p corrected -match_mode ${MODE} -auto_quit -auto_no_results -output_format csv -nowindow"

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
    rm -f ${PWD}/corrected/test.settings
done

exit ${exit_code}

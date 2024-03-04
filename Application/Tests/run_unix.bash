#!/bin/bash

PWD="$(cd $(dirname $0); pwd)"
cd $PWD

PWD="${PWD}/../../videos"
PWD="$(cd $(dirname $PWD); pwd)/$(basename $PWD)"

WPWD=${PWD}
TGRABS=trex
TREX=trex

if ! which git; then
    GIT="C:/Users/tristan/miniconda3/envs/trex/Library/bin/git.exe"
else
    GIT=git
fi

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

if ! which tgrabs; then
    if [ $(uname) == "Darwin" ]; then
        TGRABS=~/trex/Application/build/RelWithDebInfo/TRex.app/Contents/MacOS/TRex
        TREX=~/trex/Application/build/RelWithDebInfo/TRex.app/Contents/MacOS/TRex
    elif [ $(uname) == "Linux" ]; then
        TREX=~/trex/Application/build/trex
        TGRABS=~/trex/Application/build/trex
    else
        TREX=~/trex/Application/build/Release/trex
        TGRABS=~/trex/Application/build/Release/trex
    fi
fi

if [ -f "${WPWD}/average_test.png" ]; then
    # delete the average file, as to test that process as well
    rm "${WPWD}/average_test.png"
fi

CMD="${TGRABS} -d "${WPWD}" -i \"${WPWD}/test_frames/frame_%3d.jpg\" -o test -threshold 9 -average_samples 100 -averaging_method mode -meta_real_width 2304 -cm_per_pixel 1 -blob_size_ranges \"[1,10000]\" -s \"${WPWD}/test.settings\" -enable_live_tracking -auto_no_results -output_format csv -nowindow -manual_matches {} -manual_splits {} -task convert -detect_type background_subtraction"
echo "Running TGrabs... ${CMD}"
if ! { ${CMD} 2>&1; } ; then
    cat "${PWD}/tgrabs.log"
    echo "TGrabs could not be executed."
    exit_code=1
else
    echo "  Scanning files..."
    FILES=$(ls ${PWD}/data/test_fish*.csv)
    
    #if [ -z "${FILES}" ]; then
    #    echo "[ERROR] No files found."
    #    cat "${PWD}/tgrabs.log"
    #    exit_code=1
    #else
    #    f="test_fish0"
    #    echo -e "\tRunning ${GIT} --no-pager diff --word-diff --no-index -- ${PWD}/data/${f}.csv ${PWD}/compare_data_automatic/${f}.csv"
    #    echo "${PWD}/data: $(ls ${PWD}/data)"
    #    for f in ${FILES}; do
    #        f=$(basename $f .csv)

            #echo -e -n "\tChecking $f ..."
            #if ! ${GIT} --no-pager diff --word-diff --no-index -- ${PWD}/data/${f}.csv ${PWD}/compare_data_automatic/${f}.csv; then
            #    echo "FAIL"
            #    echo "[ERROR] file $f differs from baseline"
            #    exit_code=1
            #else
            #    echo 'OK'
            #fi
    #    done
    #fi

    #cat "${PWD}/tgrabs.log"
fi

rm -rf ${PWD}/data
echo ""

MODES="automatic"

for MODE in ${MODES}; do
    CMD="${TREX} -d \"${WPWD}\" -i \"${WPWD}/test\" -s \"${WPWD}/test.settings\" -p corrected -match_mode ${MODE} -auto_quit -auto_no_results -output_format csv -nowindow -manual_matches {} -manual_splits {} -task track -detect_type background_subtraction"

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
            cat "${PWD}/trex.log"
            exit_code=1
        else
            f="test_fish0"
            echo -e "\tRunning ${GIT} --no-pager diff --word-diff --no-index -- ${PWD}/corrected/data/${f}.csv ${PWD}/compare_data_${MODE}/${f}.csv"
            echo "${PWD}/corrected/data: $FILES"

            for f in ${FILES}; do
                f=$(basename $f .csv)

                echo -e -n "\tChecking $f ..."
                if ! ${GIT} --no-pager diff --word-diff --no-index -- ${PWD}/corrected/data/${f}.csv ${PWD}/compare_data_${MODE}/${f}.csv; then
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

rm -f ${PWD}/average_test.png
rm -f ${PWD}/corrected/test.results.meta

exit ${exit_code}

#!/bin/bash

PWD="$(cd $(dirname $0); pwd)"
cd $PWD

PWD="${PWD}/../../videos"
PWD="$(cd $(dirname $PWD); pwd)/$(basename $PWD)"

WPWD=${PWD}
TEST_FRAMES_DIR="${WPWD}/test_frames"
TGRABS=trex
TREX=trex
PVINFO=pvinfo
exit_code=0

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

if ! which trex; then
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

if ! command -v pvinfo >/dev/null 2>&1; then
    if [ $(uname) == "Darwin" ]; then
        PVINFO=~/trex/Application/build/RelWithDebInfo/pvinfo
    elif [ $(uname) == "Linux" ]; then
        PVINFO=~/trex/Application/build/pvinfo
    else
        PVINFO=~/trex/Application/build/Release/pvinfo
    fi
fi

if [ -f "${WPWD}/average_test.png" ]; then
    # delete the average file, as to test that process as well
    rm "${WPWD}/average_test.png"
    echo "Removing ${WPWD}/average_test.png"
fi

actual_frame_count=$(find "${TEST_FRAMES_DIR}" -maxdepth 1 -type f -name 'frame_*.jpg' | wc -l | tr -d ' ')
if [[ "${actual_frame_count}" -le 0 ]]; then
    echo "[ERROR] Expected at least one input frame in ${TEST_FRAMES_DIR}, found ${actual_frame_count}."
    exit 1
fi

FRAME_COUNT="${actual_frame_count}"
echo "Found ${FRAME_COUNT} frames in ${TEST_FRAMES_DIR}."

if [[ ! -f "${TEST_FRAMES_DIR}/frame_000.jpg" || ! -f "${TEST_FRAMES_DIR}/frame_$(printf '%03d' $((FRAME_COUNT - 1))).jpg" ]]; then
    echo "[ERROR] Expected a contiguous ${FRAME_COUNT}-frame sequence starting at frame_000.jpg."
    exit 1
fi

function compare_csv() {
  local file1="$1"
  local file2="$2"

  # Check if either file is empty
  if [[ -z "$file1" || -z "$file2" ]]; then
    return 1
  fi

  # Use comm to compare lines, ignoring the first line (header)
  if ! ${GIT} --no-pager diff --word-diff --no-index -- ${file1} ${file2} > /dev/null; then
    #echo "${file1} != ${file2}"
    return 1
  fi

  #echo "${file1} == ${file2}"
  return 0
}

function compare_csv_folder() {
  local input_folder="$1"
  local output_folder="$2"

  # Get all CSV files in the input folder
  input_files=( "$input_folder"/*.csv )

  # Get all CSV files in the output folder
  output_files=( "$output_folder"/*.csv )

  # Flag for any unmatched files
  unmatched=0

  # Loop through each file in the input folder
  for input_file in "${input_files[@]}"; do
    found_match=false

    echo -n "Checking ${input_file}..."
    # Loop through each file in the output folder
    for output_file in "${output_files[@]}"; do

      # Compare current input file with output files
      if compare_csv "$input_file" "$output_file"; then
        echo " == $(basename ${output_file[@]})!"
        #echo ${GIT} --no-pager diff --word-diff --no-index -- ${input_file} ${output_file}
        found_match=true
        break
      #else
        #echo ${GIT} --no-pager diff --word-diff --no-index -- ${input_file} ${output_file}
        #${GIT} --no-pager diff --word-diff --no-index -- ${input_file} ${output_file}
      fi
    done

    if ! $found_match; then
        unmatched=$((unmatched+1))
        echo "No match found for $input_file!"
        file1=$input_file

        # Optionally, find the closest diff match (modify this logic as needed)
        closest_diff=""
        closest_diff_file=""
        for file2 in "${output_files[@]}"; do
            diff_lines=$(${GIT} --no-pager diff --word-diff --no-index -- "$file2" "$file1" | wc -l)
            #echo "diff with $file2: $diff_lines"
            if [[ -z $closest_diff || $diff_lines -lt $closest_diff ]]; then
                closest_diff=$diff_lines
                closest_diff_file=$file2
            fi
        done

        if [[ ! -z $closest_diff ]]; then
            echo "Closest difference found with $closest_diff_file ($closest_diff)"
            ${GIT} --no-pager diff --word-diff --no-index -- ${closest_diff_file} ${input_file}
        fi
    fi
  done

  if [[ $unmatched -gt 0 ]]; then
    if [[ $closest_diff -le 12 ]]; then
        if [[ $unmatched -eq 1 ]]; then
            echo "Closest difference found with $closest_diff_file ($closest_diff) is > 0, but acceptable (likely floating point error)."
            return 0
        else
            echo "More than 1 unmatched file found. Cannot accept any differences."
            return 1
        fi
    fi
    return 1
  fi

  #echo "All CSV files in $input_folder found matches in $output_folder (ignoring names)."
  return 0
}

function check_frame_count_with_pvinfo() {
    local label="$1"
    local prefix="${2:-}"
    local pvinfo_args=(-d "${WPWD}" -i "${WPWD}/test" -quiet)
    local video_length_output=""
    local frames_output=""
    local video_length=""
    local frames=""

    if [[ -n "${prefix}" ]]; then
        pvinfo_args+=(-p "${prefix}")
    fi

    if ! video_length_output="$(${PVINFO} "${pvinfo_args[@]}" -print_parameters "[video_length]" 2>&1)"; then
        echo "[ERROR] pvinfo failed while checking ${label} video_length."
        echo "${video_length_output}"
        return 1
    fi

    video_length=$(printf '%s\n' "${video_length_output}" | awk -F'= ' '/^video_length = / {print $2; exit}' | tr -d '[:space:]')
    if [[ -z "${video_length}" ]]; then
        echo "[ERROR] Could not parse video_length from pvinfo for ${label}."
        echo "${video_length_output}"
        return 1
    fi

    if ! frames_output="$(${PVINFO} "${pvinfo_args[@]}" -plain_text 2>&1)"; then
        echo "[ERROR] pvinfo failed while checking ${label} frame count."
        echo "${frames_output}"
        return 1
    fi

    frames=$(printf '%s\n' "${frames_output}" | awk '/^frames / {print $2; exit}' | tr -d '[:space:]')
    if [[ -z "${frames}" ]]; then
        echo "[ERROR] Could not parse frames from pvinfo for ${label}."
        echo "${frames_output}"
        return 1
    fi

    if [[ "${video_length}" != "${FRAME_COUNT}" ]]; then
        echo "[ERROR] pvinfo video_length for ${label} was ${video_length}, expected ${FRAME_COUNT}."
        return 1
    fi

    if [[ "${frames}" != "${FRAME_COUNT}" ]]; then
        echo "[ERROR] pvinfo frames for ${label} was ${frames}, expected ${FRAME_COUNT}."
        return 1
    fi

    echo "  pvinfo verified ${label}: video_length=${video_length}, frames=${frames}."
    return 0
}

rm \"${WPWD}/corrected/test.settings\"
CMD="${TGRABS} -d "${WPWD}" -i \"${TEST_FRAMES_DIR}/frame_%3d.jpg\" -o test -s \"${WPWD}/test.settings\" -auto_quit -nowindow -task convert -detect_type background_subtraction -history_matching_log history_matching_tgrabs.html"
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
        cat "${PWD}/tgrabs.log"
        #ls -la ${PWD}/*
        exit_code=1
    else
        if ! compare_csv_folder "${PWD}/data" "${PWD}/compare_data_automatic"; then
            echo "FAIL"
            echo "[ERROR] corrected files differ from baseline"
            exit_code=1
        else
            echo 'OK'
            if ! check_frame_count_with_pvinfo "convert output"; then
                exit_code=1
            fi
        fi
    fi

    #cat "${PWD}/tgrabs.log"
fi

rm -rf ${PWD}/data
echo ""

MODES="automatic"

for MODE in ${MODES}; do
    CMD="${TREX} -d \"${WPWD}\" -i \"${WPWD}/test\" -s \"${WPWD}/test.settings\" -auto_quit -nowindow -task track -p corrected -match_mode ${MODE} -history_matching_log history_matching_trex.html"

    echo "Running TRex (${MODE})... ${CMD}"

    if ! { ${CMD} 2>&1; } > "${PWD}/trex.log"; then
        cat "${PWD}/trex.log"
        echo -e "\n\nTRex could not be executed."
        exit_code=1
    else
        echo "  Scanning files..."
        FILES=$(ls ${PWD}/corrected/data/test_fish*.csv)
        
        if [ -z "${FILES}" ]; then
            echo "[ERROR] No files found."
            cat "${PWD}/trex.log"
            #ls -la ${PWD}/*
            exit_code=1
        else
            #f="test_fish0"
            #echo -e "\tRunning ${GIT} --no-pager diff --word-diff --no-index -- ${PWD}/compare_data_${MODE}/${f}.csv ${PWD}/corrected/data/${f}.csv"
            #echo "${PWD}/corrected/data: $FILES"

            if ! compare_csv_folder "${PWD}/corrected/data" "${PWD}/compare_data_${MODE}"; then
                echo "FAIL"
                echo "[ERROR] corrected files differ from baseline"
                exit_code=1
            else
                echo 'OK'
                if ! check_frame_count_with_pvinfo "track output (${MODE})" "corrected"; then
                    exit_code=1
                fi
            fi

            #for f in ${FILES}; do
            #    f=$(basename $f .csv)

                #echo -e -n "\tChecking $f ..."
                #if ! ${GIT} --no-pager diff --word-diff --no-index -- ${PWD}/corrected/data/${f}.csv ${PWD}/compare_data_${MODE}/${f}.csv; then
                #    echo "FAIL"
                #    echo "[ERROR] corrected file $f differs from baseline"
                #    exit_code=1
                #else
                #    echo 'OK'
                #fi
            #done
        fi
    fi

    if [ "${exit_code}" -ne 0 ]; then
        echo "TRex (${MODE}) failed."
        cat "${PWD}/trex.log"
        # Keep outputs for artifact collection on failure.
    else
        echo "TRex (${MODE}) completed successfully."
        # Clean outputs on success to keep workspace tidy.
        rm -rf ${PWD}/corrected/data
        rm -f ${PWD}/corrected/test.settings
    fi
done

if [ "${exit_code:-0}" = "0" ]; then
  rm -f ${PWD}/average_test.png
  rm -f ${PWD}/corrected/test.results.meta
fi

exit "${exit_code:-0}"

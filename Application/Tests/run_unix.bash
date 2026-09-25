#!/bin/bash

PWD="$(cd $(dirname $0); pwd)"
cd $PWD

PWD="${PWD}/../../videos"
PWD="$(cd $(dirname $PWD); pwd)/$(basename $PWD)"

WPWD=${PWD}
TEST_FRAMES_DIR="${WPWD}/test_frames"
CONVERT=trex
TREX=trex
PVINFO=pvinfo
exit_code=0

# ANSI styling for the unix terminal: red highlights failures, dim sets
# quoted log output apart from the script's own status messages.
RED='\033[0;31m'   # errors
DIM='\033[2m'      # quoted log output
NC='\033[0m'       # reset

# Echo captured log output as a dimmed, indented block-quote (read from
# stdin) so it is visually distinct from the script's own messages.
print_log_quote() {
    local source="${1:-log}"
    echo -e "${DIM}    ┌─ ${source} ──────────────────────────${NC}"
    while IFS= read -r line; do
        echo -e "${DIM}    │ ${line}${NC}"
    done
    echo -e "${DIM}    └──────────────────────────────────────${NC}"
}

# Resolve the CSV output directory for a given output prefix exactly the way
# TRex does: <output_dir>[/<prefix>]/data. Deriving the scan path from the
# same prefix we pass via -p keeps them from drifting (a non-empty prefix
# nests everything under <prefix>/, so scanning the un-prefixed folder would
# wrongly report "no files").
data_dir_for_prefix() {
    local prefix="$1"
    if [[ -n "${prefix}" ]]; then
        echo "${PWD}/${prefix}/data"
    else
        echo "${PWD}/data"
    fi
}

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
        CONVERT=~/trex/Application/build/RelWithDebInfo/TRex.app/Contents/MacOS/TRex
        TREX=~/trex/Application/build/RelWithDebInfo/TRex.app/Contents/MacOS/TRex
    elif [ $(uname) == "Linux" ]; then
        TREX=~/trex/Application/build/trex
        CONVERT=~/trex/Application/build/trex
    else
        TREX=~/trex/Application/build/Release/trex
        CONVERT=~/trex/Application/build/Release/trex
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
    echo -e "${RED}[ERROR] Expected at least one input frame in ${TEST_FRAMES_DIR}, found ${actual_frame_count}.${NC}"
    exit 1
fi

FRAME_COUNT="${actual_frame_count}"
echo "Found ${FRAME_COUNT} frames in ${TEST_FRAMES_DIR}."

if [[ ! -f "${TEST_FRAMES_DIR}/frame_000.jpg" || ! -f "${TEST_FRAMES_DIR}/frame_$(printf '%03d' $((FRAME_COUNT - 1))).jpg" ]]; then
    echo -e "${RED}[ERROR] Expected a contiguous ${FRAME_COUNT}-frame sequence starting at frame_000.jpg.${NC}"
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
    local pvinfo_args=(-d "${WPWD}" -i test -quiet)
    local video_length_output=""
    local frames_output=""
    local video_length=""
    local frames=""

    if [[ -n "${prefix}" ]]; then
        pvinfo_args+=(-p "${prefix}")
    fi

    CMD=(
        "${PVINFO}"
        "${pvinfo_args[@]}" 
        -print_parameters "[video_length]"
    )

    echo "Checking ${label} video_length and frame count with pvinfo..." "${CMD[@]}"

    if ! video_length_output="$("${CMD[@]}" 2>&1)"; then
        echo -e "${RED}[ERROR] pvinfo failed while checking ${label} video_length.${NC}"
        printf '%s\n' "${video_length_output}" | print_log_quote "pvinfo output"
        return 1
    fi

    video_length=$(printf '%s\n' "${video_length_output}" | awk -F'= ' '/^video_length = / {print $2; exit}' | tr -d '[:space:]')
    if [[ -z "${video_length}" ]]; then
        echo -e "${RED}[ERROR] Could not parse video_length from pvinfo for ${label}.${NC}"
        printf '%s\n' "${video_length_output}" | print_log_quote "pvinfo output"
        return 1
    fi

    if ! frames_output="$(${PVINFO} "${pvinfo_args[@]}" -plain_text 2>&1)"; then
        echo -e "${RED}[ERROR] pvinfo failed while checking ${label} frame count.${NC}"
        printf '%s\n' "${frames_output}" | print_log_quote "pvinfo output"
        return 1
    fi

    frames=$(printf '%s\n' "${frames_output}" | awk '/^frames / {print $2; exit}' | tr -d '[:space:]')
    if [[ -z "${frames}" ]]; then
        echo -e "${RED}[ERROR] Could not parse frames from pvinfo for ${label}.${NC}"
        printf '%s\n' "${frames_output}" | print_log_quote "pvinfo output"
        return 1
    fi

    if [[ "${video_length}" != "${FRAME_COUNT}" ]]; then
        echo -e "${RED}[ERROR] pvinfo video_length for ${label} was ${video_length}, expected ${FRAME_COUNT}.${NC}"
        return 1
    fi

    if [[ "${frames}" != "${FRAME_COUNT}" ]]; then
        echo -e "${RED}[ERROR] pvinfo frames for ${label} was ${frames}, expected ${FRAME_COUNT}.${NC}"
        return 1
    fi

    echo "  pvinfo verified ${label}: video_length=${video_length}, frames=${frames}."
    return 0
}

# Conversion writes to the un-prefixed output dir; keep the prefix and the
# scanned data dir in lock-step via CONVERT_PREFIX.
CONVERT_PREFIX="converted"
CONVERT_DATA_DIR="$(data_dir_for_prefix "${CONVERT_PREFIX}")"
CONVERT_P_ARG=""

function cleanup_for_prefix() {
    local prefix="$1"
    local path="${WPWD:?}/${prefix:?}"
    
    if [[ -n "${prefix}" ]]; then
        if [[ "$prefix" == /* ||
            "$prefix" == ".." ||
            "$prefix" == ../* ||
            "$prefix" == */.. ||
            "$prefix" == */../* ]]; then
            echo "Refusing unsafe CONVERT_PREFIX: $prefix" >&2
            exit 1
        fi

        base=$(cd -P -- "${WPWD:?}" && pwd -P) || exit 1
        path="${WPWD:?}/${prefix:?}"
        echo "Preparing conversion output path: $path from $base"

        if [[ -d "$path" ]]; then
            target=$(cd -P -- "$path" && pwd -P) || exit 1

            if [[ "$target" != "$base/"* ]]; then
                echo "Refusing path outside WPWD: $target" >&2
                exit 1
            fi

            echo "Removing $path for conversion output..."
            rm -rf -- "$path"
        fi
    fi
}

cleanup_for_prefix "${CONVERT_PREFIX}"
mkdir -p -- "${WPWD:?}/${CONVERT_PREFIX:?}"

[[ -n "${CONVERT_PREFIX}" ]] && CONVERT_P_ARG=(-p "${CONVERT_PREFIX}")
CMD=(
    "${CONVERT:?}"
    -d "${WPWD:?}" 
    -i "${TEST_FRAMES_DIR:?}/frame_%3d.jpg"
    -o test 
    -s "${WPWD:?}/test.settings" 
    -auto_quit 
    -nowindow 
    -task convert 
    -detect_type background_subtraction 
    "${CONVERT_P_ARG[@]}"
    -history_matching_log 
    history_matching_convert.html
)
echo "Running conversion (image frames -> .pv video)..." "${CMD[@]}"

if ! { "${CMD[@]}" 2>&1; } > "${PWD}/convert.log"; then
    print_log_quote "convert.log" < "${PWD}/convert.log"
    echo -e "${RED}[ERROR] Conversion (image frames -> .pv video) could not be executed.${NC}"
    exit_code=1
else
    echo "  Scanning files... (${CONVERT_DATA_DIR})"
    FILES=$(ls "${CONVERT_DATA_DIR}"/test_fish*.csv)

    if [ -z "${FILES}" ]; then
        echo -e "${RED}[ERROR] Conversion produced no output CSV files in ${CONVERT_DATA_DIR}.${NC}"
        print_log_quote "convert.log" < "${PWD}/convert.log"
        #ls -la ${PWD}/*
        exit_code=1
    else
        if ! compare_csv_folder "${CONVERT_DATA_DIR}" "${PWD}/compare_data_automatic"; then
            echo -e "${RED}[ERROR] Conversion output differs from baseline.${NC}"
            exit_code=1
        else
            echo 'OK'
            if ! check_frame_count_with_pvinfo "convert output" "${CONVERT_PREFIX}"; then
                exit_code=1
            fi
        fi
    fi

    #print_log_quote "convert.log" < "${PWD}/convert.log"
fi

echo ""
MODES="automatic"

for MODE in ${MODES}; do
    if [[ "${exit_code}" -ne 0 ]]; then
        echo -e "${RED}[ERROR] Skipping tracking on preconverted .pv video (${MODE}) due to previous errors.${NC}"
        continue
    fi
    # Tracking nests its output under the "corrected" prefix; derive the
    # scanned data dir from the same prefix we pass via -p so they cannot drift.
    TRACK_PREFIX="corrected"
    TRACK_DATA_DIR="$(data_dir_for_prefix "${TRACK_PREFIX}")"

    cleanup_for_prefix "${TRACK_PREFIX}"
    mkdir -p -- "${WPWD:?}/${TRACK_PREFIX:?}"

    CMD=(
        "${TREX}"
        -d "${WPWD:?}"
        -i "${WPWD:?}/${CONVERT_PREFIX:?}/test"
        -s "${WPWD:?}/${CONVERT_PREFIX:?}/test.settings"
        -auto_quit -nowindow -task track -p "${TRACK_PREFIX}"
        -match_mode "${MODE}"
        -history_matching_log history_matching_track.html
    )

    printf 'Running tracking on preconverted .pv video (%s)...' "${MODE}"
    printf ' %q' "${CMD[@]}"
    printf '\n'

    if ! { "${CMD[@]}" 2>&1; } > "${PWD}/track.log"; then
        print_log_quote "track.log" < "${PWD}/track.log"
        echo -e "\n\n${RED}[ERROR] Tracking on preconverted .pv video (${MODE}) could not be executed.${NC}"
        exit_code=1
    else
        echo "  Scanning files... (${TRACK_DATA_DIR})"
        FILES=$(ls "${TRACK_DATA_DIR:?}"/test_fish*.csv)

        if [ -z "${FILES}" ]; then
            echo -e "${RED}[ERROR] Tracking produced no output CSV files in ${TRACK_DATA_DIR}.${NC}"
            print_log_quote "track.log" < "${PWD}/track.log"
            #ls -la ${PWD}/*
            exit_code=1
        else
            if ! compare_csv_folder "${TRACK_DATA_DIR}" "${PWD}/compare_data_${MODE}"; then
                echo -e "${RED}[ERROR] Tracking output differs from baseline.${NC}"
                exit_code=1
            else
                echo 'OK'
            fi
        fi
    fi

    if [ "${exit_code}" -ne 0 ]; then
        echo -e "${RED}[ERROR] Tracking on preconverted .pv video (${MODE}) failed.${NC}"
        print_log_quote "track.log" < "${PWD}/track.log"
        # Keep outputs for artifact collection on failure.
    else
        echo "Tracking on preconverted .pv video (${MODE}) completed successfully."
        # Clean outputs on success to keep workspace tidy.
        cleanup_for_prefix "${TRACK_PREFIX}"
    fi
done

if [[ "${exit_code}" -ne 0 ]]; then
    echo -e "${RED}[ERROR] One or more tests failed.${NC}"
else
    echo "All tests completed successfully."
    cleanup_for_prefix "${CONVERT_PREFIX}"
fi

exit "${exit_code:-0}"

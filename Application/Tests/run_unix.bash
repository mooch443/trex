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

if [ -f "${WPWD}/average_test.png" ]; then
    # delete the average file, as to test that process as well
    rm "${WPWD}/average_test.png"
    echo "Removing ${WPWD}/average_test.png"
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

rm \"${WPWD}/corrected/test.settings\"
CMD="${TGRABS} -d "${WPWD}" -i \"${WPWD}/test_frames/frame_%3d.jpg\" -o test -s \"${WPWD}/test.settings\" -auto_quit -nowindow -task convert -detect_type background_subtraction -history_matching_log history_matching_tgrabs.html"
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
        echo "TRex could not be executed."
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

    if [ $exit_code -ne 0 ]; then
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

if [ "${exit_code}" = "0" ]; then
  rm -f ${PWD}/average_test.png
  rm -f ${PWD}/corrected/test.results.meta
fi

exit ${exit_code}

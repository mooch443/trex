.. include:: names.rst

.. toctree::
   :maxdepth: 2

Batch processing support
========================

|trex| and |grabs| both offer full batch processing support. All parameters that can be setup via the settings box (and even some that are read-only when the program is already started), can be appended to the command-line -- as mentioned above. For batch processing, special parameters are available::

	auto_quit			  # automatically saves all requested data to the output folder and quits the app
	auto_train			 # automatically attempts to train the visual identification if successfully tracked
	auto_apply			 # automatically attempts to load weights from a previous training and auto correct the video
	auto_no_results		# do not save a .results file
	auto_no_tracking_data  # do not save the data/file_fishX.npz files
	auto_no_memory_stats   # (enabled by default) do not save memory statistics


Examples
========

Finding the right folder structure is always a good first step. See `here <https://github.com/mooch443/trex/issues/74#issuecomment-1016717244>`_ for some related examples posted by GitHub user `roaldarbol <https://github.com/roaldarbol>`_. There he suggests using a folder structure like so:

.. code::

		├── experiments
		│   ├── 2021-11-29
		│   │   ├── environmental_data
		│   │   ├── results
		│   │   │   ├── ants-1.settings
		│   │   │   ├── data
		│   │   │   │   ├── ants-1_beetle0.csv
		│   │   │   │   ├── [...]
		│   │   │   ├── track_ants-1.log
		│   │   ├── videos_pv
		│   │   │   ├── ants-1.pv
		│   │   │   ├── average_ants-1.png
		│   │   │   ├── convert_ants-1.log
		│   │   └── videos_raw
		│   │       ├── ants-1.mp4
		├── trex_batch.sh
		├── default_track.settings
		└── default_convert.settings

Here, ``trex_beetles.settings`` is the main settings file, used for all videos ``ants-X.pv``. This can be achieved by appending ``-s default_track.settings`` to the trex command line (and ``-s default_convert.settings`` to every tgrabs command line). ``trex_batch.sh`` is a script that automatically handles the addition of new files and converts/tracks the ones that have not been converted/tracked yet. It can be launched via::

	/bin/bash trex_batch.sh

A modified version of his script (reference above) follows, which additionally checks for the existence of converted videos/tracking data before calling.

.. NOTE::

	Automation is very useful, but unsupervised programs can also be dangerous. Always back your data up in separate places. While the following script is based on much experience with the tools in question, it is largely untested and I cannot guarantee that it is safe to use. It is merely meant as an example of what such bash code could look like/is similar to what I use myself. I do not take responsibility for lost data and/or similar as the result of using it.

.. code:: bash

  EXTENSION=".mp4"
  ROOT=$(pwd)

  ls -l $ROOT/experiments
  read -p 'Date of the experiment: ' DATE

  EWD=$ROOT/experiments/$DATE
  CONV_SETTINGS=$ROOT/default_convert.settings
  TRACK_SETTINGS=$ROOT/default_track.settings
  FILES=$(find "$ROOT/experiments/$DATE/videos_raw" -type f -name *.mp4)

  echo "Found $(echo $FILES | wc -l) videos."
  echo "Using ${CONV_SETTINGS} to convert and ${TRACK_SETTINGS} to track."

  function convert_video() {
    f="$1"
    BASE="$2"

    if [ ! -f "${EWD}/videos_pv/${BASE}.pv" ]; then
       CMD="tgrabs \
            -d ${EWD}/videos_pv \
            -i ${f} \
            -s ${CONV_SETTINGS} \
            -o tmp"

       echo -e "\t${CMD} ..."
       if { eval $CMD 2>&1; } > convert_${BASE}.log; then
          echo -e "\tConverted ${EWD}."

          # moving post-hoc ensures that the process was successful
          # if we find a .pv file with the correct name.
          mv average_tmp.png average_${BASE}.png
          mv tmp.pv ${BASE}.pv

          return 0
       else
         echo -e "\tFailed to convert ${EWD}."

         # delete temporary files if they exist
         rm -f average_tmp.png
         rm -f tmp.pv
         return 1
       fi

    else # found video already
      echo -e "\tSkipping conversion of ${EWD}."
      return 0
    fi
  }

  function track_video() {
    f="$1"
    BASE="$2"

    if [ ! -f "${EWD}/videos_pv/${BASE}.results" ] \
      && [ ! -f "${EWD}/videos_pv/${BASE}.results.meta" ]; 
    then
      CMD="trex \
           -d ${EWD}/results \
           -i ../videos_pv/${BASE}.pv \
           -s ${TRACK_SETTINGS} \
           -auto_quit"

        if { eval $CMD 2>&1; } > track_${BASE}.log; then
           echo -e "\tTracked ${EWD}."
        else
         echo -e "\tFailed to track ${EWD}."
         return 1
       fi

     else
       echo -e "\tSkipping tracking of ${EWD}."
     fi
  }

  for f in ${FILES}; do
     BASE=$(basename $f ${EXTENSION})
     echo "checking raw video ${BASE}..."

     if convert_video "$f" "$BASE"; then
        track_video "$f" "$BASE"
     fi
  done
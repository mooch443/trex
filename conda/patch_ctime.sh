#!/bin/bash
set -e

# Ensure the conda environment is activated (CONDA_PREFIX is set)
if [ -z "$CONDA_PREFIX" ]; then
  echo "Error: CONDA_PREFIX is not set. Please activate your conda environment."
  exit 1
fi

# Use gcc's verbose preprocessing to list include directories.
# This command prints a list of include directories between
# "#include <...> search starts here:" and "End of search list".
INCLUDE_DIRS=$(echo | $CC -x c++ -E -v - 2>&1 | awk '
  /#include <\.\.\.> search starts here:/ {flag=1; next}
  /End of search list/ {flag=0}
  flag {print $1}
')

# Debug: show the include directories that were found.
echo "${CC} include directories found:"
echo "$INCLUDE_DIRS"
echo

# Now, search through these directories for a file named "ctime"
CTIME_FILE=""
for dir in $INCLUDE_DIRS; do
  if [ -f "$dir/ctime" ]; then
    CTIME_FILE="$dir/ctime"
    break
  fi
done

if [ -z "$CTIME_FILE" ]; then
  echo "Error: <ctime> header not found in any of the include directories."
  exit 1
fi

echo "Found <ctime> file at: $CTIME_FILE"

# Create a backup of the original <ctime> file.
BACKUP_FILE="${CTIME_FILE}.bak"
cp "$CTIME_FILE" "$BACKUP_FILE"
echo "Backup saved as $BACKUP_FILE"

# Patch the file.
# We want to disable the block guarded by:
#   #if __cplusplus >= 201703L && defined(_GLIBCXX_HAVE_TIMESPEC_GET)
# by replacing that line with a line that always evaluates false.
# The following sed command does that.
sed -i 's/#if __cplusplus >= 201703L && defined(_GLIBCXX_HAVE_TIMESPEC_GET)/#if 0  \/\/ patched to disable timespec_get block/' "$CTIME_FILE"

echo "Patch applied successfully to <ctime>."
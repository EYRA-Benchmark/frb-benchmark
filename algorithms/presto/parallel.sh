#!/bin/bash
# Maximum number of jobs to run in parallel
if [[ -z $MAXJOBS ]]; then
    MAXJOBS=4
fi

echo $MAXJOBS

# Check whether input file exists
if [ ! -f "$1" ]; then
    echo "No such file: $1"
    exit 1
fi


# Ready to spawn job
function clearToSpawn
{
    local JOBCOUNT="$(jobs -r | grep -c .)"
    if [ $JOBCOUNT -lt $MAXJOBS ] ; then
        echo 1
    else
        echo 0
    fi
    return 
}

# Loop over commands
while read line;
do
  while [ `clearToSpawn` -ne 1 ] ; do
      sleep 0.5
  done
  # Run command
  echo "Running" $line
  echo $line | sh &
done < $1

echo "Waiting for child processes to finish"
wait

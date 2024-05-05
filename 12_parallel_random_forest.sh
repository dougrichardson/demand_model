#!/bin/bash -l

#PBS -P w42
#PBS -q normal
#PBS -l ncpus=24
#PBS -l mem=4GB
#PBS -l jobfs=8GB
#PBS -l walltime=00:45:00
#PBS -l wd
#PBS -l storage=gdata/w42
#PBS -e ./logs/error.txt
#PBS -o ./logs/output.txt

module load parallel
conda activate pangeo_ML

# INPUTS=/g/data/w42/dr6273/work/demand_model/parallel_inputs.txt
INPUTS=/g/data/w42/dr6273/work/demand_model/parallel_inputs_redo_isweekend.txt

# Declare an empty array. This array will store the contents of the input file.
declare -a array

# Read the file line by line
# IFS= prevents leading/trailing whitespace from being trimmed in each line, i.e. a space between columns
# -r prevents backslash escapes from being interpreted
while IFS= read -r line
do
    # Each line of the input file is appended to the array
    array+=("$line")
done < "$INPUTS" # Specify the input file here

# The printf command is used here to print each element of the array on a new line
# The array's contents are then piped to the parallel command.
printf "%s\n" "${array[@]}" | parallel -j ${PBS_NCPUS} --colsep ' ' python run_random_forest.py "{}"

# "%s\n" The %s format specifier tells printf to interpret the arguments as strings, and \n separates the elements with newlines.
# "${array[@]}" This is a special syntax for accessing all elements of an array.
# | This is the pipe operator, it enables us to send the output of one command (printf in this case) to another command (parallel in this case).
# parallel -j ${PBS_NCPUS} This runs the given command in parallel, -j specifies the number of jobs to run at the same time.
# --colsep ' ' This tells parallel to split up the inputs at each space.
# "{}" The {} symbol is replaced with an element from the input list. For you that's each line of the INPUTS file.

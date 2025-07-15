#!/bin/bash

# Put inside folder "DiCleave-main"

prefixes=("5p" "3p" "multi")
for prefix in "${prefixes[@]}"
do
    # Set mode based on prefix
    if [ "$prefix" == "5p" ]; then
        mode=5
    elif [ "$prefix" == "3p" ]; then
        mode=3
    else
        mode=multi
    fi

    total_time=0

    for i in {0..4}
    do
        echo "Running ${prefix}_${i}..."
        
        # TRAINING
        # Extract real time using `time` and capture it
        runtime=$( { time python dicleave_t.py \
            --mode ${mode} \
            --input_file ../DiCleave-data/training_${prefix}_${i}.csv \
            --data_index 34657 \
            --output_file ../DiCleave-data/training_${prefix}_${i}_model ; } 2>&1 | grep real | awk '{print $2}' )

        # Convert time to seconds (in case format is m:ss)
        minutes=$(echo $runtime | awk -F'm' '{print $1}')
        seconds=$(echo $runtime | awk -F'm' '{print $2}' | sed 's/s//')
        total_seconds=$(echo "$minutes * 60 + $seconds" | bc)

        echo "Training Runtime for ${prefix}_${i}: $total_seconds seconds"
        total_time=$(echo "$total_time + $total_seconds" | bc)

        # Testing
        runtime=$( { time python dicleave.py \
            --mode ${mode} \
            --input_file ../DiCleave-data/test_${prefix}_${i}.csv  \
            --data_index 3465 \
            --output_file ../DiCleave-data/training_${prefix}_${i}_model/model_1_result.txt \
            --model_path ../DiCleave-data/training_${prefix}_${i}_model/model_1.pt ; } 2>&1 | grep real | awk '{print $2}' )

        # Convert time to seconds (in case format is m:ss)
        minutes=$(echo $runtime | awk -F'm' '{print $1}')
        seconds=$(echo $runtime | awk -F'm' '{print $2}' | sed 's/s//')
        total_seconds=$(echo "$minutes * 60 + $seconds" | bc)

        echo "Testing Runtime for ${prefix}_${i}: $total_seconds seconds"
        total_time=$(echo "$total_time + $total_seconds" | bc)


    done

    average=$(echo "scale=3; $total_time / 5" | bc)
    echo "Average time over 5 runs: $average seconds"
    echo "-----------------------------------------"


done




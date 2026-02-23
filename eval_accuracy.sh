#!/bin/bash

# Define your 5 tasks
TASKS=(
    "Open Google Chrome and search up the weather"
    "Open the Calendar and add a meeting at 3pm"
    "Find the nearest coffee shop on Maps"
    "Send a text to 555-0199 saying Hello"
    "Adjust the screen brightness to maximum"
)

SUCCESS_COUNT=0
TOTAL_STEPS=0
TOTAL_TASKS=${#TASKS[@]}

echo "Starting evaluation on $TOTAL_TASKS tasks..."
echo "------------------------------------------"

for TASK in "${TASKS[@]}"
do
    echo -e "\n🚀 Running: $TASK"
    
    # Run python unbuffered, pipe to tee for live view, 
    # and capture the success message directly into a variable.
    # The '2>&1' ensures we see errors too.
    RESULT=$(python3 -u run.py --task "$TASK" 2>&1 | tee /dev/tty | grep "\[agent\] Task complete after")

    # Check if the RESULT variable is NOT empty
    if [[ -n "$RESULT" ]]; then
        # Extract the step count from the captured string
        STEPS=$(echo "$RESULT" | grep -oP "Task complete after \K\d+")
        echo -e "\n✅ Success: Completed in $STEPS steps."
        
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        TOTAL_STEPS=$((TOTAL_STEPS + STEPS))
    else
        echo -e "\n❌ Failure: Reached max steps or crashed."
    fi

    echo "Preparing for next task..."
    python3 reset_device.py
    echo "------------------------------------------"
done

# Calculations using bc (Standard on macOS/Linux)
ACCURACY=$(echo "scale=2; ($SUCCESS_COUNT / $TOTAL_TASKS) * 100" | bc)
if [ $SUCCESS_COUNT -gt 0 ]; then
    AVG_STEPS=$(echo "scale=2; $TOTAL_STEPS / $SUCCESS_COUNT" | bc)
else
    AVG_STEPS=0
fi

echo -e "\n=========================================="
echo "EVALUATION SUMMARY"
echo "=========================================="
echo "Accuracy:    $ACCURACY%"
echo "Avg Steps:   $AVG_STEPS (successful runs only)"
echo "=========================================="
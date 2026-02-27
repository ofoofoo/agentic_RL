#!/bin/bash

# ── Audio Recorder ────────────────────────────────────────────────────────────
TASKS_AUDIO=(
    "Record an audio clip using Audio Recorder app and save it."
    "Record an audio clip and save it with name 'meeting_notes' using Audio Recorder app."
)

# ── Clock ─────────────────────────────────────────────────────────────────────
TASKS_CLOCK=(
    "Run the stopwatch."
    "Pause the stopwatch."
    "Create a timer with 0 hours, 5 minutes, and 30 seconds. Do not start the timer."
    "Create a timer with 1 hours, 0 minutes, and 0 seconds. Do not start the timer."
)

# ── Contacts ──────────────────────────────────────────────────────────────────
TASKS_CONTACTS=(
    "Create a new contact for Jane Smith. Their number is 5559876543."
    "Create a new contact for Bob Lee. Their number is 5551112222."
    "Go to the new contact screen and enter the following details: First Name: Alice, Last Name: Chen, Phone: 5553334444, Phone Label: Mobile. Do NOT hit save."
)

# ── Simple Calendar Pro ───────────────────────────────────────────────────────
TASKS_CALENDAR=(
    "In Google Calendar, create a calendar event on 2026-03-10 at 9h with the title 'Team Meeting' and the description 'Weekly sync with the team'. The event should last for 60 mins."
    "In Google Calendar, create a calendar event for tomorrow at 14h with the title 'Doctor Appointment' and the description 'Annual checkup'. The event should last for 30 mins."
    "In Google Calendar, create a calendar event for this Friday at 18h with the title 'Dinner with family' and the description 'At the usual restaurant'. The event should last for 120 mins."
    "In Google Calendar, create a recurring calendar event titled 'Morning Run' starting on 2026-03-01 at 7h."
    "What events do I have in the next week in Google Calendar?"
    "What is my next upcoming event in Google Calendar?"
    "In Google Calendar, delete all the calendar events on 2026-03-10."
)

# ── Simple SMS Messenger ──────────────────────────────────────────────────────
TASKS_SMS=(
    "Send a text message using Messenger to 2407513192 with message: Hey, are you free this weekend?"
    "Send a text message using Messenger to 2407513192 with message: Can you send me the meeting notes?"
    "Reply to the most recent text message using Messenger with message: Thanks for letting me know, I'll be there!"
    "Resend the message I just sent to Jane Smith in Messenger."
)

# ── Settings ──────────────────────────────────────────────────────────────────
TASKS_SETTINGS=(
    "Turn bluetooth on."
    "Turn bluetooth off."
    "Turn wifi off."
    "Turn wifi on."
    "Turn brightness to the max value."
    "Turn brightness to the min value."
    "Turn off WiFi, then enable bluetooth."
    "Turn on WiFi, then open the Chrome app."
)

# ── Markor (Notes / Files) ────────────────────────────────────────────────────
TASKS_MARKOR=(
    "Create a new note in Markor named grocery_list.md with the following text: Milk, Eggs, Bread, Butter, Coffee"
    "Create a new note in Markor named meeting_agenda.md with the following text: 1. Review Q1 results 2. Plan Q2 goals 3. Team updates"
    "Create a new folder in Markor named Projects."
    "Delete the newest note in Markor."
    "Update the Markor note grocery_list.md by adding the following text, along with a new blank line before the existing content: 'Weekly Shopping'."
    "Update the content of meeting_agenda.md to 'Meeting cancelled - rescheduled for next week.' in Markor."
)

# ── Tasks App ─────────────────────────────────────────────────────────────────
TASKS_TASKS=(
    "What are my high priority tasks in Tasks app?"
    "How many tasks do I have due next week in Tasks app?"
    "What tasks do I have due 2026-03-01 in Tasks app?"
    "What incomplete tasks do I have still have to do by 2026-03-15 in Tasks app?"
)

# ── OsmAnd Maps ───────────────────────────────────────────────────────────────
TASKS_MAPS=(
    "Add a favorite location marker for Paris, France in the Maps app."
    "Add a location marker for Tokyo, Japan in the Maps app."
)

# ── Retro Music ───────────────────────────────────────────────────────────────
TASKS_MUSIC=(
    "Create a playlist in Retro Music titled 'Chill Vibes' with the following songs, in order: Song A, Song B, Song C."
)

# ── Broccoli (Recipes) ────────────────────────────────────────────────────────
TASKS_BROCCOLI=(
    "Add the following recipes into the Broccoli app: Spaghetti Carbonara (Prep: 15 min, Ingredients: pasta, eggs, bacon, parmesan)."
    "Delete all but one of any recipes in the Broccoli app that are exact duplicates, ensuring at least one instance of each unique recipe remains."
)

# ── Camera ────────────────────────────────────────────────────────────────────
TASKS_CAMERA=(
    "Take one photo."
    "Take one video."
)

# ── Category filter ───────────────────────────────────────────────────────────
CATEGORY=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --category) CATEGORY="$2"; shift ;;
    esac
    shift
done

# Default: all tasks combined
TASKS=(
    "${TASKS_AUDIO[@]}"
    "${TASKS_CLOCK[@]}"
    "${TASKS_CONTACTS[@]}"
    "${TASKS_CALENDAR[@]}"
    "${TASKS_SMS[@]}"
    "${TASKS_SETTINGS[@]}"
    "${TASKS_MARKOR[@]}"
    "${TASKS_TASKS[@]}"
    "${TASKS_MAPS[@]}"
    "${TASKS_MUSIC[@]}"
    "${TASKS_BROCCOLI[@]}"
    "${TASKS_CAMERA[@]}"
)

case "$CATEGORY" in
    audio)      TASKS=("${TASKS_AUDIO[@]}") ;;
    clock)      TASKS=("${TASKS_CLOCK[@]}") ;;
    contacts)   TASKS=("${TASKS_CONTACTS[@]}") ;;
    calendar)   TASKS=("${TASKS_CALENDAR[@]}") ;;
    sms)        TASKS=("${TASKS_SMS[@]}") ;;
    settings)   TASKS=("${TASKS_SETTINGS[@]}") ;;
    markor)     TASKS=("${TASKS_MARKOR[@]}") ;;
    tasks)      TASKS=("${TASKS_TASKS[@]}") ;;
    maps)       TASKS=("${TASKS_MAPS[@]}") ;;
    music)      TASKS=("${TASKS_MUSIC[@]}") ;;
    broccoli)   TASKS=("${TASKS_BROCCOLI[@]}") ;;
    camera)     TASKS=("${TASKS_CAMERA[@]}") ;;
    "")         ;; # all tasks
    *)          echo "Unknown category: '$CATEGORY'."
                echo "Valid: audio, clock, contacts, calendar, sms, settings, markor, tasks, maps, music, broccoli, camera"
                exit 1 ;;
esac

# ── Eval loop ─────────────────────────────────────────────────────────────────
SUCCESS_COUNT=0
TOTAL_STEPS=0
TOTAL_TASKS=${#TASKS[@]}

echo "Starting evaluation on $TOTAL_TASKS tasks... (category: ${CATEGORY:-all})"
echo "------------------------------------------"

for TASK in "${TASKS[@]}"
do
    echo -e "\n🚀 Running: $TASK"

    RESULT=$(python3 -u run.py --task "$TASK" 2>&1 | tee /dev/tty | grep "\[agent\] Task complete after")

    if [[ -n "$RESULT" ]]; then
        STEPS=$(echo "$RESULT" | sed -n 's/.*after \([0-9]*\) step.*/\1/p')
        echo -e "\n✅ Success: Completed in $STEPS steps."
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        TOTAL_STEPS=$((TOTAL_STEPS + STEPS))
    else
        echo -e "\n❌ Failure: Reached max steps or crashed."
    fi

    echo "Preparing for next task..."
    python3 reset.py
    echo "------------------------------------------"
done

# ── Summary ───────────────────────────────────────────────────────────────────
ACCURACY=$(echo "scale=2; ($SUCCESS_COUNT / $TOTAL_TASKS) * 100" | bc)
if [ $SUCCESS_COUNT -gt 0 ]; then
    AVG_STEPS=$(echo "scale=2; $TOTAL_STEPS / $SUCCESS_COUNT" | bc)
else
    AVG_STEPS=0
fi

echo -e "\n=========================================="
echo "EVALUATION SUMMARY  (category: ${CATEGORY:-all})"
echo "=========================================="
echo "Tasks run:   $TOTAL_TASKS"
echo "Successes:   $SUCCESS_COUNT"
echo "Accuracy:    $ACCURACY%"
echo "Avg Steps:   $AVG_STEPS (successful runs only)"
echo "=========================================="
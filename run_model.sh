#!/bin/bash

if [ "$1" == "train" ]; then
    echo "[BS]Training the model ..."
    PATH_TO_DATA="$2"
    PATH_TO_SAVE="$3"

    python main.py "$1" "$PATH_TO_DATA" "$PATH_TO_SAVE"
    
elif [ "$1" = "test" ]; then
    echo "[BS]Testing the model ..."
    PATH_TO_DATA="$2"
    PATH_TO_MODEL="$3"
    PATH_TO_RESULT="$4"

    python main.py "$1" "$PATH_TO_DATA" "$PATH_TO_MODEL" "$PATH_TO_RESULT" 
else
    echo "Train Usage: bash $0 train <path_to_data> <path_to_save>"
    echo "Test Usage: bash $0 test <path_to_data> <path_to_save> <path_to_result>"
    exit 1
fi
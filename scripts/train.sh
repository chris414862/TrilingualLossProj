#!/bin/bash 

# Original Author: Wei-Ning Hsu
# Augmented by: Chris Crabtree


#### Set script constants
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SCRIPT_DIR=${SCRIPT_DIR%/}
PROJECT_DIR=${SCRIPT_DIR%/*}
DATA_TR=$PROJECT_DIR"/data/TrilingualData/hdf5/metadata/trilingual_train_HDF5.json"
DATA_VAL=$PROJECT_DIR"/data/TrilingualData/hdf5/metadata/trilingual_valid_HDF5.json"
MODE="train"
TRAIN_SCRIPT="run_ResDavenet.py"

#### Process optional arguments####
# If present, we will save output to expdir/file_name.txt
log_file="" 
# If present, we will restrict gpu usage
devs_to_use=""
# # If present, will dictate which layers get quantized
# vqon=""    
# If present, will bypass argument check and directly run $TRAIN_SCRIPT
skip_arg_check=0

args=( "$@" ) # arguments in bash's array data structure
extra_args=( ) # args for python training script
part_of_prev=0
for i in $(seq 0 $# ); do 
    if [[ $part_of_prev -eq 1 ]]; then
        part_of_prev=0
        continue

    # Extract --skip-arg-check argument
    # helpful if running this script several times programatically 
    # from a different script
    elif [[ ${args[$i]} == "--skip-arg-check" ]]; then
        skip_arg_check=1

    # Extract --log file_name.txt argument
    elif [[ ${args[$i]} == "--log" ]]; then
        log_file=${args[$(($i+1))]}
        log_file=${log_file#./}
        part_of_prev=1

    # Extract --gpus #[,#[,#]*] argument
    # Example, --gpus 2,5,6 will direct the training script to only use GPU 2, 5, and 6. 
    # Options are 0 to 7 for Dr. Harwath's DGX machine
    elif [[ ${args[$i]} == "--gpus" ]]; then
        devs_to_use=${args[$(($i+1))]}
        part_of_prev=1

    # # Chris Crabtree: Collected to add commas. Not sure why this needed to be done here, but I left it.
    # elif [[ ${args[$i]} == "--vqon" ]]; then
    #     vqon=${args[$(($i+1))]}
    #     vqonarg=$(echo $vqon | sed 's/./&,/g' | sed 's/,$//g')  # insert ',' in between
    #     part_of_prev=1

    # exp-dir is collected for display purposes. Will be sent as-is to $TRAIN_SCRIPT
    elif [[ ${args[$i]} == "--exp-dir" ]]; then
        expdir=${args[$(($i+1))]}  # Example: ./exp/RDVQ_00000_01000
        expdir=${expdir#./}
        part_of_prev=1
    else
        extra_args+=( "${args[$i]}" )
    fi
done

#### Set defaults if not found
# if [[ -z "$vqon" ]]; then
#     # Chris Crabtree: preserving most of the original logic
#     vqon="00000"
#     vqonarg=$(echo $vqon | sed 's/./&,/g' | sed 's/,$//g')  # insert ',' in between
#     echo "WARNING: --vqon argument not found. "
#     echo "         Setting to $vqon"
# fi
if [[ -z "$expdir" ]]; then
    echo "WARNING: --exp-dir argument not found."
    echo "         Experiment data will be placed in $PROJECT_DIR"
    expdir=$PROJECT_DIR

fi
if [[ -z "$devs_to_use" ]]; then
    devs_to_use="0,1,2,3,4,5,6,7"
    echo "WARNING: --gpus argument not found."
    echo "         Using all available GPUs: $devs_to_use"
fi

#### Extra preprocessing
# Make sure experiment path is absolte and only included once in log file
log_file=${log_file#$expdir}
if [[ "$expdir" != /* ]]; then
    expdir=$(pwd)/"$expdir"
fi

# If experiment directory exists exit
[ -d "$expdir" ] && echo "Experiment directory '${expdir}' exists. Resolve this before running this script again" && exit -1


# Preppend experiment path to logfile path (if logfile was given)
if [[ -n "$log_file" ]]; then
    log_file=${log_file#$expdir}
    log_file="${expdir%/}/${log_file#/}"
fi


#### Report Arguments for shell script and training script
fmt_str="%-25s%s\n"
echo "----------------Shell Script Arguements-------------------"
echo "----- These arguments control program behavior "
echo "----- before $TRAIN_SCRIPT is called"
echo ""
if [[ -n "$devs_to_use" ]]; then
    printf "$fmt_str" "Using GPU(s):" "$devs_to_use"
else
    printf "$fmt_str" "Using GPU(s):" "ALL. $devs_to_use"
fi
if [[ -n "$log_file" ]]; then
    printf "$fmt_str" "Log file:" "$log_file"
else
    printf "$fmt_str" "%-25s%s\n" "Log file:" "None given. No logging will be performed"
fi

echo ""
echo "----------------$TRAIN_SCRIPT Arguments-------------------"
printf "$fmt_str" "Experiment directory:" "$expdir"
printf "$fmt_str" "Mode:" "$MODE"
printf "$fmt_str" "Training data json:" "$DATA_TR"
printf "$fmt_str" "Validation data json:" "$DATA_VAL"
echo "Remaining arguments for $TRAIN_SCRIPT: "
echo "    '${extra_args[@]}'"
echo "--------------------------------------------------------------"


#### Confirm arguments with user (unless directed to skip)
if [[ "$skip_arg_check" -ne 1 ]]; then
    printf "$0: Do you want to continue [y/n]? "
    while read answer; do
        if [[ "$answer" == "y" ]]; then
            break
        elif [[ "$answer" == "n" ]]; then
            exit 0
        else
            echo "$0: Invalid answer. Use 'y' or 'n'."
            printf "$0: Do you want to continue [y/n]? "
        fi
    done
    printf "Starting training....\n"
    echo "---------------------------------------------------------------"
fi

# Make experiment directory
mkdir "$expdir"

#### Run $TRAIN_SCRIPT
# Current script is expected to be in parent_dir($TRAIN_SCRIPT)/scripts/
run_command="python $SCRIPT_DIR/../$TRAIN_SCRIPT --mode $MODE \
            --exp-dir $expdir \
            --data-train $DATA_TR --data-val $DATA_VAL ${extra_args[@]}"

# Set available GPUs
export CUDA_VISIBLE_DEVICES=$devs_to_use  
if [[ -z "$log_file" ]]; then
    # No double quotes around $run_command bc bash will interpret  
    # that as one long word and not be able to find that command 
    $run_command 
else
    # Send all output to tee so output is displayed on terminal
    # as well as $log_file
    $run_command 2>&1 | tee "$log_file"
fi



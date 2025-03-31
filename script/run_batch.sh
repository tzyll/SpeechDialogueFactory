ROOT_PATH=`pwd -P`
export PYTHONPATH=$ROOT_PATH/src:$PYTHONPATH
SRC_PATH=$ROOT_PATH/src


# implement parameter input, map to the py script
while getopts ":c:o:i:n:l:" opt; do
  case $opt in
    c) config_path="$OPTARG"
    ;;
    o) output_dir="$OPTARG"
    ;;
    i) input_prompt_file="$OPTARG"
    ;;
    n) num_dialogues_per_prompt="$OPTARG"
    ;;
    l) dialogue_language="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
        exit 1
    ;;
  esac
done
# check if the parameters are set

echo "config_path: $config_path"
echo "output_dir: $output_dir"
echo "input_prompt_file: $input_prompt_file"
echo "num_dialogues_per_prompt: $num_dialogues_per_prompt"
echo "dialogue_language: $dialogue_language"


python $SRC_PATH/speech_dialogue_factory.py \
--sdf_config ${config_path} \
--output_dir ${output_dir} \
--input_prompt_file ${input_prompt_file} \
--num_dialogues_per_prompt ${num_dialogues_per_prompt} \
--dialogue_language ${dialogue_language} \
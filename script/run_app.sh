ROOT_PATH=`pwd -P`
export PYTHONPATH=$ROOT_PATH/src:$PYTHONPATH
SRC_PATH=$ROOT_PATH/src
SCRIPT_PATH=$ROOT_PATH/script
export GRADIO_SERVER_NAME="0.0.0.0"
export GRADIO_SERVER_PORT=7860
export GRADIO_DEBUG=1
export GRADIO_TEMP_DIR=$ROOT_PATH/tmp
export GRADIO_ROOT_PATH=$ROOT_PATH

BIN=$SRC_PATH/app/app_main.py


python $BIN --sdf_config ./configs/sdf_config_app_oai.json
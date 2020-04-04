#!/usr/bin/python3

import sys
import os
import argparse
import configparser
import json


# Default values
DEFAULT_CONFIG_PATH = "config.ini"
ALADDIN_VERSION = "0.1" # A version number used to check for compatibility

# Default config name
DEFAULT_CONFIG_NAME = 'DEFAULT'

# Parameter names in config file
CFG_IN_SECTION = 'ConfigInName'
CFG_OUT_SECTION = 'ConfigOutName'
CFG_CLADES = 'CladePaths'
CFG_MSA = 'MSAPaths'
CFG_MODEL_SPEC = 'ModelSpecificationPath'
CFG_MODEL_WEIGHTS = 'ModelWeightsPath'
CFG_SHOULD_TRAIN = 'TrainingMode'

# Parameter names in cmd-line
PARAMETER_NAMES = {
    CFG_IN_SECTION: "load_config",
    CFG_OUT_SECTION: "save_config",
    CFG_CLADES: "clade",
    CFG_MSA: "input_files",
    CFG_MODEL_SPEC: "model_spec",
    CFG_MODEL_WEIGHTS: "model_weights",
    CFG_SHOULD_TRAIN: 'train',
}

def file_exists(arg):
    if not os.path.isfile(arg):
        raise argparse.ArgumentTypeError(f"The file {arg} does not exist!")
    return arg



def main():

    # Shorten notation
    pn = PARAMETER_NAMES

    # If a config file should be loaded 
    # do it now and use the loaded data
    # as default data for the respective 
    # parameters
    config_loader = argparse.ArgumentParser(add_help=False)
    config_loader.add_argument("--load-config",
            dest="load_config",
            metavar="CONFIG_PATH",
            nargs="?",
            type=str,
            const=DEFAULT_CONFIG_NAME)
    config_info,_ = config_loader.parse_known_args()
    config_info = vars(config_info)
    # Flag if a config file should be loaded
    config_present = config_info[pn[CFG_IN_SECTION]] is not None
    loaded_parameters = {}

    if config_present:
        config = configparser.ConfigParser()
        if os.path.isfile(DEFAULT_CONFIG_PATH):
            with open(DEFAULT_CONFIG_PATH, 'r') as configfile:
                config.read_file(configfile)
                i = config_info[pn[CFG_IN_SECTION]]

                # Clades
                if config.has_option(i, CFG_CLADES):
                    loaded_parameters[CFG_CLADES] = json.loads(config[i][CFG_CLADES])
                # MSAs
                if config.has_option(i, CFG_MSA):
                    loaded_parameters[CFG_MSA] = json.loads(config[i][CFG_MSA])
                # Should Train
                if config.has_option(i, CFG_SHOULD_TRAIN):
                    loaded_parameters[CFG_SHOULD_TRAIN] = bool(config[i][CFG_SHOULD_TRAIN])
                # Model specification path
                if config.has_option(i, CFG_MODEL_SPEC):
                    loaded_parameters[CFG_MODEL_SPEC] = config[i][CFG_MODEL_SPEC]
                # Model weights path
                if config.has_option(i, CFG_MODEL_WEIGHTS):
                    loaded_parameters[CFG_MODEL_WEIGHTS] = config[i][CFG_MODEL_WEIGHTS]

    EXAMPLE_USAGES = "Example usage:...."

    ## Setup command line options and help
    parser = argparse.ArgumentParser(
            description='Evaluates MSA\'s given a tree.',
            epilog=EXAMPLE_USAGES
    )



    parser.add_argument("--load-config",
            dest="load_config",
            metavar="CONFIG_PATH",
            nargs="?",
            type=str,
            const=DEFAULT_CONFIG_NAME,
            help=f"Command line parameters previously saved into the configuration file '{DEFAULT_CONFIG_PATH}' with name {DEFAULT_CONFIG_NAME} will be loaded. If CONFIG_PATH is provided the configuration entry CONFIG_PATH will be loaded instead.")



    # Tree Specification
    # TODO: Check that the given paths are valid
    parser.add_argument("-c", "--clade",
            metavar=("CLADE_FILE_1", "CLADE_FILE_2"),
            dest="clade",
            nargs='*' if config_present else '+',
            default=loaded_parameters[CFG_CLADES] if config_present else None,
            type=file_exists,
            help="Clade file(s) in Newick (.nwk) format",
            required=not config_present)

    # MSA Files
    # TODO: Check that the given paths are valid
    parser.add_argument("-i", "--input", 
            metavar=("INPUT_FILE_1", "INPUT_FILE_2"),
            type=file_exists, 
            default=loaded_parameters[CFG_MSA] if config_present else None,
            nargs="*" if config_present else '+', 
            help="Input file(s) in FASTA (.fs) format or in Tensorflow Record (.tfrecord.gz) format", 
            required=not config_present, 
            dest="input_files")

    # Tensorflow model that is compatible 
    # with aladdin input and output spec.
    # TODO: Check thet the given paths are valid
    parser.add_argument("--model_spec", 
            metavar="SPEC_FILE", 
            nargs=1,
            help="Tensorflow SavedModel-file compatible with aladdin input and output specification",
            type=str)

    # TCMC parameters a.k.a rate matrices and pi
    # TODO: Check that the given paths are valid
    parser.add_argument("--model_weights", 
            metavar="WEIGHTS_FILE", 
            type=str, 
            nargs=1, 
            help="Tensorflow weights-file compatible with the given model specification 'model_spec'")
    
    # 
    parser.add_argument("-t", "--train",
            dest="train", 
            action="store_true",
            help="Start a training run instead of an evaluation")

    # TODO: Check that the given path is valid
    parser.add_argument("--save-config",
            dest="save_config",
            metavar="CONFIG_NAME",
            nargs="?",
            const=DEFAULT_CONFIG_NAME,
            help=f"Saves the current command line parameters into the configuration file '{DEFAULT_CONFIG_PATH}' as the new set of default parameters. If CONFIG_NAME is provided the configuration file will be written to CONFIG_NAME instead.")


    # Start evaluating the given parameters
    args = vars(parser.parse_args())

    # TODO: Remove dummy print
    print(f"Sucessfully loaded parameters: {args}")

    # TODO: Evaluate arguments

    if args[pn[CFG_OUT_SECTION]]:

        # Read the current configs, if present
        config = configparser.ConfigParser()
        if os.path.isfile(DEFAULT_CONFIG_PATH):
            with open(DEFAULT_CONFIG_PATH, "r") as configfile:
                config.read_file(configfile)


        # Modify the wanted config entry
        with open(DEFAULT_CONFIG_PATH, "w") as configfile:
            i = args[pn[CFG_OUT_SECTION]]
            config[i] = {}

            # Mandatory parameters
            config[i][CFG_CLADES] = json.dumps(args[pn[CFG_CLADES]])
            config[i][CFG_MSA] = json.dumps(args[pn[CFG_MSA]])

            # Optional on-demand parameters
            model_spec = args[pn[CFG_MODEL_SPEC]]
            if model_spec:
                config[i][CFG_MODEL_SPEC] = model_spec

            model_weights = args[pn[CFG_MODEL_WEIGHTS]]
            if model_weights:
                config[i][CFG_MODEL_WEIGHTS] = model_weights
            
            should_train = args[pn[CFG_SHOULD_TRAIN]]
            if should_train:
                config[i][CFG_SHOULD_TRAIN] = str(True)

            config.write(configfile)

if __name__ == "__main__":
    main()

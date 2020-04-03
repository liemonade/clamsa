#!/usr/bin/python3

import os
import argparse
import configparser
import json


# Default values
DEFAULT_CONFIG_PATH = "config.ini"
ALADDIN_VERSION = "0.1" # A version number used to check for compatibility

# Default config name
DEFAULT_CONFIG_NAME = 'DEFAULT'

# Parameter names
PARAMETER_NAMES = {
    "ConfigInName": "load_config",
    "ConfigOutName": "save_config",
    "CladePaths": "clade",
    "MSAPaths": "input_files",
    "ModelSpecificationPath": "model_spec",
    "ModelWeightsPath": "model_weights",
    'TrainingMode': 'train',
}

def file_exists(arg):
    if not os.path.isfile(arg):
        raise argparse.ArgumentTypeError(f"The file {arg} does not exist!")
    return arg

def main():

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
            nargs="+",
            default=False,
            type=file_exists,
            help="Clade file(s) in Newick (.nwk) format",
            required=True)

    # MSA Files
    # TODO: Check that the given paths are valid
    parser.add_argument("-i", "--input", 
            metavar=("INPUT_FILE_1", "INPUT_FILE_2"),
            type=file_exists, 
            nargs="+", 
            help="Input file(s) in FASTA (.fs) format or in Tensorflow Record (.tfrecord.gz) format", 
            required=True, 
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

    # Shorten notation
    pn = PARAMETER_NAMES
    print(args)

    if args[pn['ConfigInName']]:
        # TODO: Implement config load feature
        print("loading config...")

    # TODO: Evaluate arguments

    if args[pn['ConfigOutName']]:

        # Read the current configs
        config = configparser.ConfigParser()
        with open(DEFAULT_CONFIG_PATH, "r") as configfile:
            config.read_file(configfile)


        # Modify the wanted config entry
        with open(DEFAULT_CONFIG_PATH, "w") as configfile:
            i = args[pn['ConfigOutName']]
            config[i] = {}

            # Mandatory parameters
            config[i]['CladePaths'] = json.dumps(args[pn['CladePaths']])
            config[i]['MSAPaths'] = json.dumps(args[pn['MSAPaths']])

            # Optional on-demand parameters
            model_spec = args[pn['ModelSpecificationPath']]
            if model_spec:
                config[i]['ModelSpecificationPath'] = model_spec

            model_weights = args[pn['ModelWeightsPath']]
            if model_weights:
                config[i]['ModelWeightsPath'] = model_weights
            
            should_train = args[pn['TrainingMode']]
            if should_train:
                config[i]['TrainingMode'] = 'True'

            config.write(configfile)

if __name__ == "__main__":
    main()

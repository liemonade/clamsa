#!/usr/bin/python3

import argparse

def main():

    DEFAULT_CONFIG_PATH = "config/aladdin.ini"

    ## Setup command line options and help
    parser = argparse.ArgumentParser(
            description='Evaluates MSA\'s given a tree.',
            epilog='Example usage:...'
    )



    parser.add_argument("--load-config",
            dest="load_config",
            metavar="CONFIG_PATH",
            nargs="?",
            help=f"Command line parameters previously saved into the configuration file '{DEFAULT_CONFIG_PATH}' will be loaded. If CONFIG_PATH is provided the configuration file CONFIG_PATH will be loaded instead.")


    # Tree Specification
    parser.add_argument("-c", "--clade",
            metavar=("CLADE_FILE_1", "CLADE_FILE_2"),
            dest="clade",
            nargs="+",
            help="Clade file(s) in Newick (.nwk) format",
            required=True)

    # MSA Files
    parser.add_argument("-i", "--input", 
            metavar=("INPUT_FILE_1", "INPUT_FILE_2"),
            type=str, 
            nargs="+", 
            help="Input file(s) in FASTA (.fs) format or in Tensorflow Record (.tfrecord.gz) format", 
            required=True, 
            dest="input_files")

    # Tensorflow model that is compatible 
    # with aladdin input and output spec.
    parser.add_argument("--model_spec", 
            metavar="SPEC_FILE", 
            nargs=1,
            help="Tensorflow SavedModel-file compatible with aladdin input and output specification",
            type=str)

    # TCMC parameters a.k.a rate matrices and pi
    parser.add_argument("--model_weights", 
            metavar="WEIGHTS_FILE", 
            type=str, 
            nargs=1, 
            help="Tensorflow weights-file compatible with the given model specification 'model_spec'")
    
    # 
    parser.add_argument("-t", "--train",
            dest="accumulate", 
            action="store_true",
            help="Start a training run instead of an evaluation")

    parser.add_argument("--save-config",
            dest="save_config",
            metavar="CONFIG_PATH",
            nargs="?",
            help=f"Saves the current command line parameters into the configuration file '{DEFAULT_CONFIG_PATH}'. If CONFIG_PATH is provided the configuration file will be written to CONFIG_PATH instead.")


    args = parser.parse_args()
    print(args)

if __name__ == "__main__":
    main()

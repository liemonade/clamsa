#!/usr/bin/env python3

import sys
import os
import argparse
import configparser
import json
import numbers
import newick
from pathlib import Path

import utilities.msa_converter as mc


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

def folder_exists_and_is_writable(arg):
    if not os.path.isdir(arg) or not os.access(arg, os.W_OK):
        raise argparse.ArgumentTypeError(f"The folder {arg} does not exist or is not writable!")
    return arg

def folder_is_writable_if_exists(arg):
    if arg == None:
        return arg
    if not os.path.isdir(arg) or not os.access(arg, os.W_OK):
        raise argparse.ArgumentTypeError(f"The folder {arg} does not exist or is not writable!")
    return arg

def is_valid_split(arg):
    try:
        splits = json.loads(arg)
        if not isinstance(splits, dict):
            argparse.ArgumentTypeError(f'The provided split {arg} does not represent a dictionairy!')

        num_minus_one = 0
        for split in splits:

            if not isinstance(splits[split], numbers.Number):
                raise argparse.ArgumentTypeError(f'The provided value "{splits[split]}" for the split "{split}" is not a number!')

            if splits[split] == -1:
                num_minus_one = num_minus_one + 1

        if num_minus_one > 1:
            raise argparse.ArgumentTypeError(f'More than one -1 is provided as a value of the splits!')

        return splits
    except ValueError:
        raise argparse.ArgumentTypeError(f'The provided split "{arg}" is not a valid JSON string!')


class Aladdin(object):

    def __init__(self):
        parser = argparse.ArgumentParser(
            usage = '''aladdin.py <command> [<args>]

Use one of the following commands:
   convert    Create an MSA dataset ready for usage in aladdin
   train      Train aladdin on an MSA dataset with given models
   evaluate   Infer likelihood for any given MSA in a dataset to be an exon
''')
        parser.add_argument('command', help='Subcommand to run')

        # parse the command
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        
        getattr(self, args.command)()

    def convert(self):
        parser = argparse.ArgumentParser(
                description='Convert an input multiple sequence alignment dataset to be used by aladdin.')

        parser.add_argument('in_type', 
                choices=['augustus', 'fasta', 'phylocsf'],
                metavar='INPUT_TYPE',
                help='Choose which type of input file(s) should be converted. Supported are: {augustus, fasta, phylocsf}')

        parser.add_argument('input_files',
            metavar="INPUT_FILE",
            nargs='+',
            type=file_exists, 
            help="Input file(s) in .out(.gz) format from AUGUSTUS, in FASTA (.fs) format or a (.zip) file from PhyloCSF")

        parser.add_argument('--tf_out_dir',
                metavar='OUTPUT_FOLDER',
                help='Folder in which the converted MSA database should be stored. By default the folder "msa/" is used.',
                type = folder_is_writable_if_exists)
        
        
        parser.add_argument('--basename',
                metavar = 'BASENAME',
                help = 'The base name of the output files to be generated. By default a concatination of the input files is used.')

        parser.add_argument('--to_phylocsf',
                help='Specifies that the MSA database should be converted to a format compatible with training and evaluating in PhyloCSF.',
                action='store_true')

        parser.add_argument('--write_nexus',
                metavar = 'NEX_FILENAME',
                help = 'A sample of positive alignments are concatenated and converted to a NEXUS format that can be used directly by MrBayes to create a tree.')

        parser.add_argument('--nexus_sample_size',
                metavar = 'N',
                help = 'The sample size (=number of alignments) of the nexus output. The sample is taken uniformly from among all positive alignments in random order.',
                type = int,
                default = 3)

        parser.add_argument('--splits', 
                help='The imported MSA database will be splitted into the specified pieces. SPLITS_JSON is assumed to be a a dictionairy in JSON notation. The keys are used in conjunction with the base name to specify an output path. The values are assumed to be either positive integers or floating point numbers between zero and one. In the former case up to this number of examples will be stored in the respective split. In the latter case the number will be treated as a percentage number and the respective fraction of the data will be stored in the split. A single "-1" is allowed as a value and all remaining entries will be stored in the respective split.',
                metavar='SPLITS_JSON',
                type=is_valid_split)

        # TODO: implement newick check
        parser.add_argument('--clades', 
                help='Provide a paths CLADES to clade file(s) in Newick (.nwk) format. The species found in the input file(s) are assumed to be contained in the leave set of exactly one these clades. If so, the sequences will be aligned in the particular order specified in the clade. The names of the species in the clade(s) and in the input file(s) need to coincide.',
                metavar='CLADES',
                type=file_exists,
                nargs='+')

        parser.add_argument('--margin_width',
                help='Whether the input MSAs are padded by a MARGIN_WIDTH necleotides on both sides.',
                metavar='MARGIN_WIDTH',
                type=int,
                default=0)

        parser.add_argument('--ratio_neg_to_pos',
                help = 'Undersample the negative samples (Model ID 0) or positive examples (Model ID 1) of the input file(s) to achieve a ratio of RATIO negative per positive example.',
                metavar = 'RATIO',
                type = float)

        parser.add_argument('--use_codons', 
                help = 'The MSAs will be exported as codon-aligned codon sequences instead of nucleotide alignments.',
                action = 'store_true')

        parser.add_argument('--use_compression',
                help = 'Whether the output files should be compressed using GZIP or not. By default compression is used.',
                action = 'store_false')
        
        parser.add_argument('--subsample_lengths',
                help = 'Negative examples of overrepresented length are undersampled so that the length distributions of positives and negatives are similar. Defaults to false.',
                action = 'store_true')

        parser.add_argument('--verbose',
                help = 'Whether some logging of the import and export should be performed.',
                action = 'store_true')


        parser.add_argument('--split_models',
                help = 'Whether the dataset should be divided into multiple chunks depending on the models of the sequences. By default no split is performed. Say one wants to split models 0 and 1 then one may achive this by "--split_models 0 1".',
                type = int,
                nargs = '+')

        # ignore the initial args specifying the command
        args = parser.parse_args(sys.argv[2:])

        if args.basename == None:
            args.basename = '_'.join(Path(p).stem for p in args.input_files)

        if args.in_type == 'augustus':
            T, species = mc.import_augustus_training_file(args.input_files,
                                                          reference_clades = args.clades,
                                                          margin_width = args.margin_width)

        if args.in_type == 'phylocsf':
            T, species = mc.import_phylocsf_training_file(args.input_files,
                                                          reference_clades = args.clades,
                                                          margin_width = args.margin_width)
        
        # harmonize the length distributions if requested
        if args.subsample_lengths:
            T = mc.subsample_lengths(T, args.use_codons)
        
        # achieve the requested ratio of negatives to positives
        if args.ratio_neg_to_pos:
            T = mc.subsample_labels(T, args.ratio_neg_to_pos)

        # write NEXUS format for tree construction
        if args.write_nexus:
            mc.export_nexus(T, species, nex_fname = args.write_nexus,
                            n = args.nexus_sample_size, use_codons = args.use_codons)

        # store MSAs in tfrecords, if requested and existing
        if args.tf_out_dir and len(T) > 0:
            num_skipped = mc.persist_as_tfrecord(dataset=T,
                    out_dir = args.tf_out_dir,
                    basename = args.basename,
                    species = species,
                    splits = args.splits,
                    split_models = args.split_models,
                    use_codons = args.use_codons,
                    use_compression = args.use_compression,
                    verbose = args.verbose)

            print(f'The datasets have sucessfully been saved in tfrecord files. {num_skipped} entries have been skipped due to beeing too short.')

def main():

    Aladdin()
    exit(0)


    # TODO: Old concept code, remove!
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

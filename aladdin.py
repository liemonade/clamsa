#!/usr/bin/env python3

import sys
import os
import argparse, textwrap
import configparser
import json
import numbers
import newick
from pathlib import Path
import pandas as pd
import collections

import utilities.msa_converter as mc
import utilities.model_evaluation as me


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
        splits = json.loads(arg, object_pairs_hook=OrderedDict) # so the input order is kept
        if not isinstance(splits, dict):
            argparse.ArgumentTypeError(f'The provided split {arg} does not represent a dictionary!')
        for split in splits:
            if not isinstance(splits[split], numbers.Number):
                raise argparse.ArgumentTypeError(f'The provided value "{splits[split]}" for the split "{split}" is not a number!')
        return splits
    except ValueError:
        raise argparse.ArgumentTypeError(f'The provided split "{arg}" is not a valid JSON string!')

def is_valid_json(arg):
    try:
        obj = json.loads(arg)
        return obj
    except ValueError:
        raise argparse.ArgumentTypeError(f'The provided split "{arg}" is not a valid JSON string!')

class Aladdin(object):

    def __init__(self):
        parser = argparse.ArgumentParser(
            description='''
Discriminative evolutionary classification of alignments.
Authors: Darvin Mertsch, Mario Stanke
''',
            usage = '''aladdin.py <command> [<args>]

Use one of the following commands:
   convert    Create an MSA dataset ready for usage in aladdin
   train      Train aladdin on an MSA dataset with given models
   predict    Infer probability for an MSA to be a coding exon
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
                help = 'The base name of the output files to be generated. By default a concatination of the input files is used.',
        )

        parser.add_argument('--phylocsf_out_dir',
                help = 'Specifies that the MSA database should (also) be converted to PhyloCSF format.',
                type = folder_is_writable_if_exists)

        parser.add_argument('--write_nexus',
                metavar = 'NEX_FILENAME',
                help = 'A sample of positive alignments are concatenated and converted to a NEXUS format that can be used directly by MrBayes to create a tree.')

        parser.add_argument('--nexus_sample_size',
                metavar = 'N',
                help = 'The sample size (=number of alignments) of the nexus output. The sample is taken uniformly from among all positive alignments in random order.',
                type = int,
                default = 3)

        parser.add_argument('--splits', 
                help='The imported MSA database will be splitted into the specified pieces. SPLITS_JSON is assumed to be a a dictionairy in JSON notation. The keys are used in conjunction with the base name to specify an output path. The values are assumed to be either positive integers or floating point numbers between zero and one. In the former case up to this number of examples will be stored in the respective split. In the latter case the number will be treated as a percentage number and the respective fraction of the data will be stored in the split. A value of -1 specifies that the remaining entries are distributed among the splits of negative size. All (filtered) examples are used in this case.',
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
        parser.add_argument('--subsample_lengths_relax',
                help = 'Factor for length subsampling probability of negatives. If > 1, the subsampling delivers more data but the negative length distribution fits not as closely that of the positives. Default=1.0', type=float, default=1.0)
        
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
            
        if args.in_type == 'fasta':
            T, species = mc.import_fasta_training_file(args.input_files,
                                                       reference_clades = args.clades,
                                                       margin_width = args.margin_width)

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
            T = mc.subsample_lengths(T, args.use_codons, relax=args.subsample_lengths_relax)
        
        # achieve the requested ratio of negatives to positives
        if args.ratio_neg_to_pos:
            T = mc.subsample_labels(T, args.ratio_neg_to_pos)

        print ("Number of filtered alignments available to be written: ", len(T))
        
        if len(T) > 0:
            # write NEXUS format for tree construction
            if args.write_nexus:
                mc.export_nexus(T, species, nex_fname = args.write_nexus,
                                n = args.nexus_sample_size, use_codons = args.use_codons)
            
            # compute actual split sizes: how many alignments to write in test, validation, training sets
            splits, split_models, split_bins, n_wanted \
            = mc.preprocess_export(T, species,
                                   args.splits,
                                   args.split_models,
                                   args.use_codons, 
                                   args.verbose)
            
            # store MSAs in tfrecords, if requested
            if args.tf_out_dir:
                num_skipped = mc.persist_as_tfrecord(T,
                        args.tf_out_dir,
                        args.basename,
                        species,
                        splits, split_models, split_bins, n_wanted,
                        use_codons = args.use_codons,
                        use_compression = args.use_compression,
                        verbose = args.verbose)

                print(f'The datasets have sucessfully been saved in tfrecord files.')
            
            # store MSAs in PhyloCSF format, if requested
            if args.phylocsf_out_dir:
                mc.write_phylocsf(T,
                        args.phylocsf_out_dir,
                        args.basename,
                        species,
                        splits, split_models, split_bins, n_wanted,
                        use_codons = args.use_codons)
                
                print(f'The datasets have sucessfully been saved in PhyloCSF files.')

                
    def train(self):
        
        parser = argparse.ArgumentParser(
                description='Train a series of models and hyperparameter configurations on an input multiple sequence alignment dataset generated by aladdin.')

        parser.add_argument('input_dir', 
                            metavar='INPUT_DIR',
                            help='Folder in which the converted MSA database should be stored. By default the folder "msa/" is used.',
                            type = folder_is_writable_if_exists,
        )
        
        parser.add_argument('--basenames',
                            metavar = 'BASENAMES',
                            help = 'The base name of the input files.',
                            nargs='+',
        )
        
        
        parser.add_argument('--clades', 
                            help='Path(s) to the clades files (.nwk files, with branch lengths) used in the converting process. CAUTION: The same ordering as in the converting process must be used!',
                            metavar='CLADES',
                            type=file_exists,
                            nargs='+',
        )
        
        parser.add_argument('--merge_behaviour',
                            metavar='MERGE_BEHAVIOUR',
                            help='In which ratio the respective splits for each basename shall be merged. The possible modes are: "evenly", "columns", "sequences", "w_1 ... w_n". Where "evenly" means all basenames have the same weight. In the mode "columns" the total number of alignment columns for each basename is counted and the weights are adjusted accordingly. In mode "sequences" the total number of sequences for each basename is counted and the weights are adjusted accordingly. Lastly a set of costum weights can be given directly. Default is "evenly".',
                            nargs='+',
        )
        
        
        parser.add_argument('--split_specifications', 
                            help='TODO: Write help',
                            metavar='SPLIT_SPECIFICATIONS',
                            type=is_valid_json,
        )

        parser.add_argument('--used_codons', 
                            help = 'The MSAs were exported as codon-aligned codon sequences instead of nucleotide alignments.',
                            action = 'store_true',
        )
        
        parser.add_argument('--model_hyperparameters', 
                            help='TODO: Write help',
                            metavar='MODEL_HYPERPARAMETERS',
                            type=is_valid_json,
        )
        
        
        parser.add_argument('--model_training_callbacks', 
                            help='TODO: Write help',
                            metavar='MODEL_TRAINING_CALLBACKS',
                            type=is_valid_json,
        )
        
        
        parser.add_argument('--batch_size',
                            help='Number of MSAs per training batch.',
                            metavar='BATCH_SIZE',
                            type=int,
                            default=30,
        )
        
        
        parser.add_argument('--batches_per_epoch',
                            help='Number of training batches in each epoch.',
                            metavar='BATCHES_PER_EPOCH',
                            type=int,
                            default=100,
        )
        
        

        parser.add_argument('--epochs',
                            help='Number of epochs per hyperparameter configuration.',
                            metavar='BATCH_SIZE',
                            type=int,
                            default=40,
        )
        
        
        parser.add_argument('--save_model_weights', 
                            help = 'Whether the weights of the best performing models shall be saved.',
                            action = 'store_true',
        )
        
        
        parser.add_argument('--log_basedir',
                            metavar='LOG_BASEDIR',
                            help='Folder in which the Tensorboard training logs should be stored. Defaults to "./logs/"',
                            type = folder_is_writable_if_exists,
        )
        
        parser.add_argument('--saved_weights_basedir',
                            metavar='SAVED_WEIGHTS_BASEDIR',
                            help='Folder in which the weights for the best performing models should be stored. Defaults "./saved_weights/"',
                            type = folder_is_writable_if_exists,
        )
        
        
        
        parser.add_argument('--verbose', 
                            help = 'Whether training informtion should be printed to console. All ',
                            action = 'store_true',
        )
        
        
        # ignore the initial args specifying the command
        args = parser.parse_args(sys.argv[2:])
        
        
        from utilities.training import train_models
        
        
        train_models(args.input_dir, 
                     args.basenames,
                     args.clades,
                     args.merge_behaviour,
                     args.split_specifications,
                     args.used_codons,
                     args.model_hyperparameters,
                     args.model_training_callbacks,
                     args.batch_size,
                     args.batches_per_epoch,
                     args.epochs,
                     args.save_model_weights,
                     args.log_basedir,
                     args.saved_weights_basedir,
                     args.verbose,
        )
        
    
    def predict(self):
        
        
        parser = argparse.ArgumentParser(
            description='Predict the class of multiple sequence alignments with one or more models.',
            formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument('in_type',
                choices=['fasta'],
                metavar='INPUT_TYPE',
                help='Specif the input file type. Supported are: {fasta}',
        )
        

        parser.add_argument('input', 
                            metavar='INPUT',
                            help='A space separated list of paths to text files containing themselves paths to MSA files of the chosen input type.\nEach MSA file contains a single alignment.',
                            type=file_exists,
                            nargs='+',
        )
        
        parser.add_argument('--clades', 
                            help='Path(s) to the clades files (.nwk files, with branch lengths) used in the converting process.\nCAUTION: The same ordering as in the converting process must be used!',
                            metavar='CLADES',
                            type=file_exists,
                            nargs='+',
        )

        
        
        parser.add_argument('--use_codons', 
                            help = 'The MSAs will be exported as codon-aligned codon sequences instead of nucleotide alignments.',
                            action = 'store_true',
        )
        
        
        
        parser.add_argument('--batch_size',
                            help='Number of MSAs to evaluate per computation step.\nHigher batch sizes increase the speed of evaluation, though require more RAM / VRAM in the case of CPU / GPU evaluation.',
                            metavar='BATCH_SIZE',
                            type=int,
                            default=30,
        )
        
        parser.add_argument('--log_basedir',
                            metavar='LOG_BASEDIR',
                            help='Folder in which the Tensorboard training logs are stored. Defaults to "./logs/"',
                            type = folder_is_writable_if_exists,
        )
        
        
        parser.add_argument('--saved_weights_basedir',
                            metavar='SAVED_WEIGHTS_BASEDIR',
                            help='Folder in which the weights for the best performing models are stored.\nDefaults "./saved_weights/"',
                            type = folder_is_writable_if_exists,
        )
        
        
        parser.add_argument('--model_ids',
                            metavar='MODEL_IDS',
                            help='Trial-IDs of trained models residing in the LOG_BASEDIR folder with weights stored in SAVED_WEIGHTS_BASEDIR.',
                            type=is_valid_json,
                           )
        
        
        parser.add_argument('--out_csv', 
                           metavar='OUT_CSV',
                           help='Output file name for the *.csv file containing the predictions.',
        )

        parser.add_argument('--name_translation', 
                            help='''Path to a file that contains an optional translation table.
The sequence names in the fasta MSA input are translated to clade ids as used in the clade .nwk files.
In the tab-separated 2-column file, the first column holds the seqence name, the second the taxon id.
The first column cannot contain duplicates. A space separated list of paths is allowed, too.
Example:
dm       dmel
droAna   dana
dm3.chr1 dmel''',
                            metavar='TRANSTBL',
                            type=file_exists,
                            nargs='+',
        )


        
        # ignore the initial args specifying the command
        args = parser.parse_args(sys.argv[2:])
        
        
        
        # import the list of fasta file paths
        fasta_paths = []
        for fl in args.input:
            with open(fl) as f:
                fasta_paths.extend(f.read().splitlines())

        model_ids = collections.OrderedDict(args.model_ids) # to fix the models order as in the command-line argument

        # read name->taxon_id translation tables into dictionary if specified
        trans_dict = {}
        if not args.name_translation is None:
            for trfn in args.name_translation:
                with open(trfn) as f:
                    for line in f.read().splitlines():
                        a = line.split('\t')
                        if len(a) != 2:
                            raise Exception(f"Translation file {trfn} contains an error in line {line}. Must have 2 tab-separated fields.")
                        (fasta_name, taxon_id) = a
                        if fasta_name in trans_dict and trans_dict[fasta_name] != taxon_id:
                            raise Exception(f"Translation file {trfn} contains conflicting duplicates: {fasta_name} -> {trans_dict[fasta_name]}, {taxon_id}")
                        trans_dict[fasta_name] = taxon_id

        

        preds = me.predict_on_fasta_files(trial_ids=args.model_ids,
                                          saved_weights_dir=args.saved_weights_basedir,
                                          log_dir=args.log_basedir,
                                          clades=args.clades,
                                          fasta_paths = fasta_paths,
                                          use_codons = args.use_codons,
                                          batch_size = args.batch_size,
                                          trans_dict = trans_dict,
        )

        # construct a dataframe from the predictions
        df = pd.DataFrame.from_dict(preds)

        if not args.out_csv is None:
            df.to_csv(args.out_csv, sep='\t',
                      float_format = '%.4f', # output precision
                      index=False,
                      header = True,
                      mode = 'w' 
            ) 
            
def main():
    Aladdin()
    exit(0)

if __name__ == "__main__":
    main()

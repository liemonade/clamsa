import gzip
import regex as re
import random
import numpy as np
from Bio import SeqIO
import sys
import math
import numbers
from contextlib import ExitStack
import time
import os
from tqdm import tqdm
import newick
from collections import Counter
import itertools
import json
import zipfile
import io


class MSA(object):
    def __init__(self, model = None, chromosome_id = None, start_index = None, end_index = None, is_on_plus_strand = False, frame = 0, spec_ids = [], offsets = [], sequences = []):
        self.model = model
        self.chromosome_id = chromosome_id
        self.start_index = start_index
        self.end_index = end_index
        self.is_on_plus_strand = is_on_plus_strand
        self.frame = frame
        self.spec_ids = spec_ids
        self.offsets = offsets
        self.sequences = sequences
        self._updated_sequences = True
        self._coded_sequences = None

    @property
    def coded_sequences(self, alphabet="acgt"):

        # Lazy loading
        if not self._updated_sequences:
            return self._coded_sequences

        # whether the sequences and coding alphabet shall be flipped
        inv_coef = -1 if not self.is_on_plus_strand else 1

        # translated alphabet as indices of (inversed) alphabet
        translated_alphabet = dict(zip( alphabet, range(len(alphabet))[::inv_coef] ))

        # view the sequences as a numpy array
        ca = np.array([list(S[::inv_coef]) for S in self.sequences])

        # translate the list of sequences and convert it to a numpy matrix
        # non-alphabet characters are replaced by -1
        self._coded_sequences = np.vectorize(lambda c: translated_alphabet.get(c, -1))(ca)

        # Update lazy loading
        self._updated_sequences = False


        return self._coded_sequences

    @property
    def codon_alignment(self, alphabet='acgt', gap_symbols='-'):

        # size of a codon in characters
        c = 3

        # the list of sequences that should be codon aligned
        sequences = self.sequences

        # TODO: ask whether this is the right behavior
        if not self.is_on_plus_strand:
            rev_alphabet = alphabet[::-1]
            tbl = str.maketrans(alphabet, rev_alphabet)
            sequences = [s[::-1].translate(tbl) for s in sequences]


        return tuple_alignment(sequences, gap_symbols, frame=self.frame, tuple_length=c)

    @property
    def sequences(self):
        return self._sequences
    @sequences.setter
    def sequences(self, value):
        self._sequences = value
        self._updated_sequences = True

    def __str__(self):
        return f"{{\n\tmodel: {self.model},\n\tchromosome_id: {self.chromosome_id},\n\tstart_index: {self.start_index},\n\tend_index: {self.end_index},\n\tis_on_plus_strand: {self.is_on_plus_strand},\n\tframe: {self.frame},\n\tspec_ids: {self.spec_ids},\n\toffsets: {self.offsets},\n\tsequences: {self.sequences},\n\tcoded_sequences: {self.coded_sequences},\n\tcodon_alignment: {self.codon_alignment}\n}}"
    



def tuple_alignment(sequences, gap_symbols='-', frame=0, tuple_length=3):
    """
    Align a list of string sequences to tuples of a fixed length with respect to a set of gap symbols.

    Args:
        sequences (List[str]) The list of sequences
        gap_symbols (str): A string containing all symbols to be treated as gap-symbols
        frame (int): Ignore the first `(tuple_length - frame) % tuple_length` symbols found
        tuple_length (int): Length of the tuples to be gathered

    Returns:
        List[str]: The list of tuple strings of the wanted length. The first gap-symbols `gap_symbols[0]` is used to align these (see the example).
    Example:
        * Two codons are considered aligned, when all 3 of their bases are aligned with each other.
        * Note that not all bases of an ExonCandidate need be aligned.
        * example input (all ECs agree in phase at both boundaries)
        *        
        *                 a|c - - t t|g a t|g t c|g a t|a a 
        *                 a|c - - c t|a a - - - c|a n c|a g
        *                 g|c g - t|t g a|- g t c|g a c|a a
        *                 a|c g t|t t g|a t - t|c g a|c - a
        *                 a|c g - t|t g a|t g t|t g a|- a a
        *                   ^                       ^
        * firstCodonBase    |                       | lastCodonBase (for last species)
        * example output: (stop codons are excluded for singe and terminal exons)
        *                 - - -|c t t|- - -|g t c|- - -|g a t
        *                 - - -|c c t|- - -|- - -|- - -|a n c
        *                 c g t|- - -|t g a|g t c|- - -|g a c
        *                 - - -|- - -|- - -|- - -|c g a|- - -
        *                 c g t|- - -|t g a|- - -|t g a|- - -
        *
        Reproduce via 
               S = ['ac--ttgatgtcgataa',
                    'ac--ctaa---cancag',
                    'acg-ttga-gtcgacaa',
                    'acgtttgat-tcgac-a',
                    'acg-ttgatgttga-aa']
               print(tuple_alignment(S, frame=2))
    """
    # shorten notation
    S = sequences


    # number of entries missing until the completion of the framing tuple
    frame_comp = (tuple_length - frame) % tuple_length

    # pattern to support frames, i.e. skipping the first `frame_comp` tuple entries at line start
    frame_pattern = '(?:(?:^' + f'[^{gap_symbols}]'.join([f'[{gap_symbols}]*' for i in range(frame_comp+1)]) + f')|[{gap_symbols}]*)\K'
    
    # generate pattern that recognizes tuples of the given length and that ignores gap symbols
    tuple_pattern = f'[{gap_symbols}]*'.join([f'([^{gap_symbols}])' for i in range(tuple_length)])
    tuple_re = re.compile(frame_pattern + tuple_pattern)
    
    # for each sequence find the tuples of indicies 
    T = [set(tuple(m.span(i+1)[0] for i in range(tuple_length)) for m in tuple_re.finditer(s)) for s in S]
    
    # flatten T to a list and count how often each multiindex is encountered
    occ = Counter( list(itertools.chain.from_iterable([list(t) for t in T])) )
    
    # find those multiindices that are in more than one sequence and sort them lexicographically
    I = sorted([i for i in occ if occ[i] > 1])

    # calculate a matrix with `len(S)` rows and `len(I)` columns.
    #   if the j-th multindex is present in sequence `i` the entry `(i,j)` of the matrix will be 
    #   a substring of length `tuple_length` corresponding to the characters at the positions 
    #   specified by the `j`-th multiindex
    #
    #   otherwise the prime gap_symbol (`gap_symbol[0]`) will be used as a filler
    missing_entry = gap_symbols[0] * tuple_length
    entry_func = lambda i,j: ''.join([S[i][a] for a in I[j]]) if I[j] in T[i] else missing_entry
    ta_matrix = np.vectorize(entry_func)(np.arange(len(S))[:,None],np.arange(len(I)))
    
    # join the rows to get a list of tuple alignment sequences
    ta = [''.join(row) for row in ta_matrix]
    return ta





def leave_order(path, use_alternatives=False):
    """
        Find the leave names in a Newick file and return them in the order specified by the file
        
        Args:
            path (str): Path to the Newick file
        
        Returns:
            List[str]: Leave names encountered in the file
    """
    
    # regex that finds leave nodes in a newick string
    # these are precisely those nodes which do not have children
    # i.e. to the left of the node is either a '(' or ',' character
    # or the beginning of the line
    leave_regex = re.compile('(?:^|[(,])([a-zA-Z]*)[:](?:(?:[0-9]*[.])?[0-9]+)')
    
    with open(path, 'r') as fp:
        
        nwk_string = fp.readline()
        
        matches = leave_regex.findall(nwk_string)
        
        alt_path = path + ".alt"
        
        if use_alternatives and os.path.isfile(alt_path):
            with open(alt_path) as alt_file:
                alt = json.load(alt_file)
                
                # check for valid input 
                assert isinstance(alt, dict), f"Alternatives in {alt_path} is no dictionary!"
                for i in alt:
                    assert isinstance(alt[i], list), f"Alternative for {i} in {alt_path} is not a list!"
                    for entry in alt[i]:
                        assert isinstance(entry, str), f"Alternative {alt[i][j]} for {i} in {alt_path} is not a string!"
                
                matches = [set([matches[i]] + alt[matches[i]]) for i in range(len(matches))]
            
        return matches

    




def import_augustus_training_file(paths, undersample_neg_by_factor = 1., alphabet=['a', 'c', 'g', 't'], reference_clades=None, margin_width=0):
    """ Imports the training files generated by augustus. This method is tied to the specific
     implementation of GeneMSA::getAllOEMsas and GeneMSA::getMsa.
     
    Args:
        paths (List[str]): Location of the file(s) generated by augustus (typically denoted *.out or *.out.gz)
        undersample_neg_by_factor (float): take any negative sample only with probability 1 / undersample_neg_by_factor, set to 1.0 to use all negative examples
        alphabet (List[str]): Alphabet in which the sequences are written
        reference_clades (newick.Node): Root of a reference clade. The given order of species in this tree will be used in the input file(s). 
        margin_width (int): Width of flanking region around sequences
    
    Returns:
        List[MSA]: Training examples read from the file(s).
        int: Number of unique species encountered in the file(s).
    
    """
    
    
    # entries already read
    training_data = []
    
    # counts number of unique species encountered the file
    num_species = 0

    # The species encountered in the file(s).
    # If clades are specified the leave species will be imported.
    species = [leave_order(c) for c in reference_clades] if reference_clades != None else []


    # total number of bytes to be read
    total_bytes = sum([os.path.getsize(x) for x in paths])

    # Status bar for the reading process
    pbar = tqdm(total=total_bytes, desc="Parsing AUGUSTUS file(s)", unit='b', unit_scale=True)


    for p in range(len(paths)):

        path = paths[p]

        with gzip.open(path, 'rt') if path.endswith('.gz') else open(path, 'r') as f:

            # species encountered in the header of the file
            encountered_species = {}

            # if a reference clade is specified we need to translate the indices
            # else just use the order specified in the file
            spec_reorder = {}
            
            # Regex Pattern recognizing lines generated by GeneMSA::getMsa
            slice_pattern = re.compile("^[0-9]+\\t[0-9]+\\t\\t[acgt\-]+")
            
            # whether the current sequence should be skipped due to 
            # a undersample roll
            skip_entry = False

            line = f.readline()
            bytes_read = f.tell()

            # if the header is completely read, i.e. all species are read
            header_read = False
            
            while line:
                
                # parse the lines generated by GeneMSA::getAllOEMsas
                if line.startswith("y="):

                    if not header_read:

                        if reference_clades != None:
                            e = set(encountered_species.values())
                            C = [set(c) for c in species]
                            esubC = [e <= c for c in C]
                            num_parents = sum(esubC)

                            if num_parents == 0:
                                raise Exception(f'The species {list(encountered_species.values())} found in "{path}" are not fully included in any of the given clades.')
                            if num_parents > 1:
                                parent_clades = [reference_clades[i] for i in range(len(C)) if e <= C[i]]
                                raise Exception(f'The species {list(encountered_species.values())} found in "{path}" are included in all of the clades {parent_clades}.')
                            # index of the clade to use
                            i = esubC.index(True)
                            ref = species[i]
                            spec_reorder = {j:(i,ref.index(encountered_species[j])) for j in encountered_species}
                        else:
                            spec_reorder = {j:(p,j) for j in encountered_species}
                            species.append(list(encountered_species.values()))

                        header_read = True
                    
                    # decide whether the upcoming entry should be skipped
                    skip_entry = line[2]=='0' and random.random() > 1. / undersample_neg_by_factor
                    
                    if skip_entry:
                        continue
                    
                    oe_data = line.split("\t")
                    
                    msa = MSA(
                            model = int(oe_data[0][2]),
                            chromosome_id = oe_data[2], 
                            start_index = int(oe_data[3]),
                            end_index = int(oe_data[4]),
                            is_on_plus_strand = (oe_data[5] == '+'),
                            frame = int(oe_data[6][0]),
                            spec_ids = [],
                            offsets = [],
                            sequences = []
                    )
                    
                    training_data.append(msa)
                
                # parse the lines generated by GeneMSA::getMsa
                elif slice_pattern.match(line) and not skip_entry:
                    slice_data = line.split("\t")
                    entry = training_data[-1]

                    entry.spec_ids.append(spec_reorder[ int(slice_data[0]) ])
                    entry.offsets.append(int(slice_data[1]))
                    padded_sequence = slice_data[3][:-1]
                    sequence = padded_sequence[margin_width:-margin_width] if margin_width > 0 else padded_sequence
                    entry.sequences.append(sequence)

                    
                # retrieve the number of species
                elif line.startswith("species ") and not skip_entry:
                    spec_line = line[len('species '):].split('\t')
                    specid = int(spec_line[0])
                    spec_name = spec_line[1].strip()
                    encountered_species[specid] = spec_name

                line = f.readline()
                pbar.update(f.tell() - bytes_read)
                bytes_read = f.tell()

    return training_data, species


def import_phylocsf_training_file(paths, undersample_neg_by_factor = 1., reference_clades=None, margin_width=0):
    """ Imports archives of training files generated by PhyloCSF in zip format
     
    Args:
        paths (List[str]): Location of the file(s) generated by augustus (typically denoted *.out or *.out.gz)
        undersample_neg_by_factor (float): take any negative sample only with probability 1 / undersample_neg_by_factor, set to 1.0 to use all negative examples
        alphabet (List[str]): Alphabet in which the sequences are written
        reference_clades (newick.Node): Root of a reference clade. The given order of species in this tree will be used in the input file(s). 
        margin_width (int): Width of flanking region around sequences
    
    Returns:
        List[MSA]: Training examples read from the file(s).
        List[List[str]]: Unique species configurations either encountered or given by reference.
    
    """

    # TODO: support either the non-clades case or make reference_clades a required parameter
    species = [leave_order(c,use_alternatives=True) for c in reference_clades] if reference_clades != None else []

    training_data = []

    with ExitStack() as stack:

        #file_indices = [(split_name, model, ) for ]
        phylo_files = [zipfile.ZipFile(path, 'r') for path in paths]
        # read the filenames of all fasta files inside these archives
        fastas = [[p.getinfo(n) for n in p.namelist() if n.endswith('.mfa')] for p in phylo_files]

        # total number of bytes to be read
        total_bytes = sum([sum([f.compress_size for f in fastas[i]]) for i in range(len(phylo_files))])

        # Status bar for the reading process
        pbar = tqdm(total=total_bytes, desc="Parsing PhyloCSF file(s)", unit='b', unit_scale=True)

        for i, phylo_file in enumerate(phylo_files):
            for j, fasta in enumerate(fastas[i]):
                
                model = 0 if 'control' in fasta.filename else 1
                
                # decide whether the upcoming entry should be skipped
                skip_entry = model==0 and random.random() > 1. / undersample_neg_by_factor

                if skip_entry:
                    continue
                
                with io.TextIOWrapper(phylo_file.open(fasta), encoding="utf-8") as fafile:

                    entries = [rec for rec in SeqIO.parse(fafile, "fasta")]
                    
                    # parse the species names
                    spec_in_file = [e.id.split('|')[0] for e in entries]
                    
                    # compare them with the given references
                    ref_ids = [[(r,i) for r in range(len(species))  for i in range(len(species[r])) if s in species[r][i] ] for s in spec_in_file]

                    # check if these are contained in exactly one reference clade
                    n_refs = [len(x) for x in ref_ids]

                    if 0 == min(n_refs) or max(n_refs) > 1:
                        continue

                    ref_ids = [x[0] for x in ref_ids]

                    if len(set(r for (r,i) in ref_ids)) > 1:
                        continue

                    # the first entry of the fasta file has the header informations
                    header_fields = entries[0].id.split("|")
                    
                    # read the sequences and trim them if wanted
                    sequences = [str(rec.seq).lower() for rec in entries]
                    sequences = sequences[margin_width:-margin_width] if margin_width > 0 else sequences

                    msa = MSA(
                            model = model,
                            chromosome_id = None, 
                            start_index = None,
                            end_index = None,
                            is_on_plus_strand = True if len(header_fields) < 5 or header_fields[4] != 'revcomp' else False,
                            frame = int(header_fields[2][-1]),
                            spec_ids = ref_ids,
                            offsets = [],
                            sequences = sequences
                    )


                    training_data.append(msa)

                    pbar.update(fasta.compress_size)

    return training_data, species



def write_example(imodel, num_species, iconfigurations, leaf_configuration, tfwriter, alphabet_card = 4, verbose = False):
    """Write one-hot encoded MSA as an entry into a  Tensorflow-Records file.
    
    Args:
        imodel (int): Model ID. In our convention: 0 - non-coding region, 1 - coding region
        num_species (int): Total number of species
        iconfiguration (List[int]): Indices of species occuring in the MSA.
        leaf_configuration (numpy.array): MSA
        tfwriter (TFRecordWriter): Target to which the example shall be written.
    """
    
    # Infer the length of the sequences
    sequence_length = leaf_configuration.shape[1]

    s = alphabet_card
    
    
    # one-hot encoding of characters
    leaf_onehot = np.ones((num_species, sequence_length, s), dtype = np.int32)
    leaf_onehot[iconfigurations, ...] = \
        ((np.arange(s) == leaf_configuration[:, :, None]) | # an actual codon
         (None == leaf_configuration[:, :, None]) # contains at least one unknown character
        ).astype(int)
    p_leaf_onehot = leaf_onehot.tostring()

    if verbose:
        np.set_printoptions(threshold = np.inf)
        print(f"model: {imodel}")
        print(f"iconfiguration: {iconfigurations}")
        print(f"sequence_length: {sequence_length}")
        print(f"leaf_onehot.shape: {leaf_onehot.shape}")
        print(f"leaf_onehot[iconfigurations[1],...]: {leaf_onehot[iconfigurations[1], ...]}")

    # TODO loeschen
    iconfigurations = np.arange(num_species).tolist()

    # put the bytes in context_list and feature_list
    ## save imodel and iconfigurations in context list 
    context_lists = tf.train.Features(feature = {
        'model': tf.train.Feature(int64_list = tf.train.Int64List(value = [imodel])),
        'configurations': tf.train.Feature(int64_list = tf.train.Int64List(value = iconfigurations)),
        'sequence_length': tf.train.Feature(int64_list = tf.train.Int64List(value = [sequence_length])),
    })

    ## save p_leaf_onehot as a one element sequence in feature_lists
    leaf_onehot_list_pickle = [tf.train.Feature(bytes_list = tf.train.BytesList(value = [p_leaf_onehot]))]

    feature_lists = tf.train.FeatureLists(feature_list = {
        'sequence_onehot': tf.train.FeatureList(feature = leaf_onehot_list_pickle)
    })

    # create the SequenceExample
    SeqEx = tf.train.SequenceExample(
        context = context_lists,
        feature_lists = feature_lists
        )
    SeqEx_serialized = SeqEx.SerializeToString()

    tfwriter.write(SeqEx_serialized)

    
# TODO: Implement use_codon feature
def persist_as_tfrecord(dataset, out_dir, basename, num_species, splits=None, split_models=None, use_codons=False, use_compression=True, verbose=False):

    # Importing Tensorflow takes a while.
    # Therefore to not slow down the rest 
    # of the script it is only imported once used
    import tensorflow as tf

    options = tf.io.TFRecordOptions(compression_type = 'GZIP') if use_compression else None

    # Prepare for iteration
    splits = splits if splits != None else {None: 1.0}
    split_models = split_models if split_models != None else [None]

    # Check whether splits are valid
    split_values_are_numbers = all([ isinstance(x, numbers.Number) for x in list(splits.values()) ])

    if not split_values_are_numbers:
        raise ValueError("The values of the dict `splits` must be numbers. But it is given by {splits}")

    # Convert relative numbers to absolute numbers
    n_data = len(dataset)
    to_total = lambda x: int(abs(x) * n_data) if isinstance(x, float) else abs(x)
    split_totals = np.array([to_total(x) for x in list(splits.values())])

    n_wanted = sum(split_totals)

    # If the wanted number exceeds the total of the dataset
    # rescale accordindly
    if n_wanted > n_data:
        split_totals = split_totals * (n_data / n_wanted)

    # The upper bound indices used to decide where to write the `i`-th entry
    split_bins = np.cumsum(split_totals)

    # Generate target file name based on chosen split and model
    target_path = lambda split_name, model: f"{out_dir}{basename}{'-'+split_name if split_name != None else ''}{'-m'+str(model) if model != None else ''}.tfrecord{'.gz' if use_compression else ''}"


    with ExitStack() as stack:
    
        #file_indices = [(split_name, model, ) for ]
        tfwriters = [[
                stack.enter_context(tf.io.TFRecordWriter(target_path(sn,m), options = options))
            for m in split_models] for sn in splits]

        n_written = np.zeros_like(split_totals)

        for i in tqdm(range(n_wanted), desc="Writing dataset", unit=" MSA"):

            msa = dataset[i]

            # TODO: Ignore sequences with only one species
            #if T[i]["S"].shape[0] < 2 or pmap[i] < 0:
            #    continue
            # context features to be saved
            iconfigurations = msa.spec_ids

            leaf_configuration = msa.coded_sequences
                    
            # TODO: Max and min length
            
            model = msa.model

            s = np.digitize(i, split_bins)
            m = split_models.index(model) if model in split_models else 0
            
            tfwriter = tfwriters[s][m]

            write_example(model, num_species, iconfigurations, leaf_configuration, tfwriter, alphabet_card = 4,  verbose = verbose)
                
            if verbose:
                print(f"leaf_configuration[1,...]: {leaf_configuration.shape} {leaf_configuration[1,...]}")
                ichar_test = np.random.randint(0, sequence_length)
                print("ichar_test:", ichar_test)
                print(f"Sample of the conversion:")
                print(f"\tModel: {imodel}")
                print(f"\tSequence Length: {sequence_length}")
                print(f"\tConfiguration: {iconfigurations}")
                print(f"\tPosition {ichar_test} character of the sequence: {leaf_configuration[:, ichar_test]}")


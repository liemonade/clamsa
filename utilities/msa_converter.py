#!/usr/bin/env python3
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
import matplotlib.pyplot as plt
from . import onehot_tuple_encoder as ote

stop_codons = {"taa", "tag", "tga"}

class MSA(object):
    def __init__(self, model = None, chromosome_id = None, start_index = None, end_index = None,
                 is_on_plus_strand = False, frame = 0, spec_ids = [], offsets = [], sequences = []):
        self.model = model # label, class, e.g. y=1 for coding, y=0 for non-coding
        self.chromosome_id = chromosome_id
        self.start_index = start_index # chromosomal position
        self.end_index = end_index
        self.is_on_plus_strand = is_on_plus_strand
        self.frame = frame
        self.spec_ids = spec_ids
        self.offsets = offsets
        self.sequences = sequences
        self._updated_sequences = True
        self._coded_sequences = None
        self.in_frame_stop = False


    @property
    def coded_sequences(self, alphabet = "acgt"):
        # lazy loading
        if not self._updated_sequences:
            return self._coded_sequences

        # whether the sequences and coding alphabet shall be flipped
        inv_coef = -1 if not self.is_on_plus_strand else 1

        # translated alphabet as indices of (inversed) alphabet
        translated_alphabet = dict(zip( alphabet, range(len(alphabet))[::inv_coef] ))

        # view the sequences as a numpy array
        ca = [S[::inv_coef] for S in self.sequences]

        # translate the list of sequences and convert it to a numpy matrix
        # non-alphabet characters are replaced by -1
        self._coded_sequences = ote.OnehotTupleEncoder.encode(ca, tuple_length=1, use_bucket_alphabet=False)

        # update lazy loading
        self._updated_sequences = False

        return self._coded_sequences

    @property
    def codon_aligned_sequences(self, alphabet='acgt', gap_symbols='-'):
        # size of a codon in characters
        c = 3

        # the list of sequences that should be codon aligned
        sequences = self.sequences

        # reverse complement if not on plus strand
        if not self.is_on_plus_strand:
            rev_alphabet = alphabet[::-1]
            tbl = str.maketrans(alphabet, rev_alphabet)
            sequences = [s[::-1].translate(tbl) for s in sequences]

        cali, self.in_frame_stop = tuple_alignment(sequences, gap_symbols, frame = self.frame, tuple_length = c)
        return cali

    @property
    def coded_codon_aligned_sequences(self, alphabet='acgt', gap_symbols = '-'):
        ca = self.codon_aligned_sequences
        return ote.OnehotTupleEncoder.encode(ca,tuple_length=3, use_bucket_alphabet=False)

    @property
    def sequences(self):
        return self._sequences

    def alilen(self, use_codons = False):
        # TODO: is somewhere checked that all rows have the same length?
        if (self._coded_sequences is not None):
            length = len(self._coded_sequences[0])
        else:
            length = len(self._sequences[0])
            if use_codons:
                length = int(length / 3) # may differ from the number of codon columns
        return length

    @sequences.setter
    def sequences(self, value):
        self._sequences = value
        self._updated_sequences = True

    def __str__(self):
        return f"{{\n\tmodel: {self.model},\n\tchromosome_id: {self.chromosome_id},\n\tstart_index: {self.start_index},\n\tend_index: {self.end_index},\n\tis_on_plus_strand: {self.is_on_plus_strand},\n\tframe: {self.frame},\n\tspec_ids: {self.spec_ids},\n\toffsets: {self.offsets},\n\tsequences: {self.sequences},\n\tcoded_sequences: {self.coded_sequences},\n\tcodon_aligned_sequences: {self.codon_aligned_sequences}\n}}"
    



def tuple_alignment(sequences, gap_symbols='-', frame = 0, tuple_length = 3):
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
        * example output: (stop codons are excluded for single and terminal exons)
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
    frame_comp = frame # this is the GGF frame definition, was before: frame_comp = (tuple_length - frame) % tuple_length

    # pattern to support frames, i.e. skipping the first `frame_comp` tuple entries at line start
    frame_pattern = '(?:(?:^' + f'[^{gap_symbols}]'.join([f'[{gap_symbols}]*' for i in range(frame_comp+1)]) + f')|[{gap_symbols}]*)\K'
    
    # generate pattern that recognizes tuples of the given length and that ignores gap symbols
    tuple_pattern = f'[{gap_symbols}]*'.join([f'([^{gap_symbols}])' for i in range(tuple_length)])
    tuple_re = re.compile(frame_pattern + tuple_pattern)
    
    # for each sequence find the tuples of indices 
    T = [set(tuple(m.span(i+1)[0] for i in range(tuple_length)) for m in tuple_re.finditer(s)) for s in S]
    
    # flatten T to a list and count how often each multiindex is encountered
    occ = Counter( list(itertools.chain.from_iterable([list(t) for t in T])) )
    
    # find those multiindices that are in more than one sequence and sort them lexicographically
    I = sorted([i for i in occ if occ[i] > 1])

    # trivial case: there is nothing to align
    if len(I) == 0:
        return ['' for s in S]

    # calculate a matrix with `len(S)` rows and `len(I)` columns.
    #   if the j-th multindex is present in sequence `i` the entry `(i,j)` of the matrix will be 
    #   a substring of length `tuple_length` corresponding to the characters at the positions 
    #   specified by the `j`-th multiindex
    #
    #   otherwise the prime gap_symbol (`gap_symbol[0]`) will be used as a filler
    missing_entry = gap_symbols[0] * tuple_length
    entry_func = lambda i,j: ''.join([S[i][a] for a in I[j]]) if I[j] in T[i] else missing_entry
    ta_matrix = np.vectorize(entry_func)(np.arange(len(S))[:,None],np.arange(len(I)))
    
    # remove last column if it contains a stop codon (happens for single and terminal exons)
    stops_in_lastcol = set(ta_matrix[:,-1]) & stop_codons
    if stops_in_lastcol:
        ta_matrix = ta_matrix[:, 0:-1]
        
    # check if there is an in-frame stop codon somewhere
    stops_anywhere = False
    for i in range(ta_matrix.shape[1]):
        stops_anywhere = stops_anywhere or bool(set(ta_matrix[:,i]) & stop_codons)

    # join the rows to get a list of tuple alignment sequences
    ta = [''.join(row) for row in ta_matrix]
    return ta, stops_anywhere



def leaf_order(path, use_alternatives=False):
    """
        Find the leaf names in a Newick file and return them in the order specified by the file
        
        Args:
            path (str): Path to the Newick file
            use_alternatives (bool): TODO
        Returns:
            List[str]: Leaf names encountered in the file
    """
    
    # regex that finds leave nodes in a newick string
    # these are precisely those nodes which do not have children
    # i.e. to the left of the node is either a '(' or ',' character
    # or the beginning of the line
    leave_regex = re.compile('(?:^|[(,])([a-zA-Z0-9]*)[:](?:(?:[0-9]*[.])?[0-9]+)')
    
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


def import_fasta_training_file(paths, undersample_neg_by_factor = 1., reference_clades = None, margin_width = 0):
    """ Imports the training files in fasta format.
    Args:
        paths (List[str]): Location of the file(s) 
        undersample_neg_by_factor (float): take any negative sample only with probability 1 / undersample_neg_by_factor,
                                           set to 1.0 to use all negative examples
        reference_clades (newick.Node): Root of a reference clade. The given order of species in this tree will be used in the input file(s). 
        margin_width (int): Width of flanking region around sequences
    
    Returns:
        List[MSA]: Training examples read from the file(s).
        List[List[str]]: Unique species configurations either encountered or given by reference.

    """
    
    training_data = []
    
    # If clades are specified the leave species will be imported.
    species = [leaf_order(c) for c in reference_clades] if reference_clades != None else []
    
    # total number of bytes to be read
    total_bytes = sum([os.path.getsize(x) for x in paths])

    # Status bar for the reading process
    pbar = tqdm(total = total_bytes, desc = "Parsing FASTA file(s)", unit = 'b', unit_scale = True)
    
    # TODO: fasta files in .gz format
    fasta_files = [open(path, 'r') for path in paths]
    
    for i, fasta in enumerate(fasta_files):
        
        bytes_read = fasta.tell()
        
        model = 0 if 'control' in fasta.name else 1
                
        # decide whether the upcoming entry should be skipped
        skip_entry = model==0 and random.random() > 1. / undersample_neg_by_factor

        if skip_entry:
            continue

        entries = [rec for rec in SeqIO.parse(fasta, "fasta")]
                    
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
                    
        # read the sequences and trim them if wanted
        sequences = [str(rec.seq).lower() for rec in entries]
        sequences = sequences[margin_width:-margin_width] if margin_width > 0 else sequences

        msa = MSA(
                model = model,
                chromosome_id = None, 
                start_index = None,
                end_index = None,
                is_on_plus_strand = True, # TODO: right strand
                frame = 0, # TODO: right frame
                spec_ids = ref_ids,
                offsets = [],
                sequences = sequences
        )
        
        training_data.append(msa)
        
        pbar.update(fasta.tell() - bytes_read)
        bytes_read = fasta.tell()
        
    return training_data, species


def import_augustus_training_file(paths, undersample_neg_by_factor = 1., alphabet=['a', 'c', 'g', 't'],
                                  reference_clades = None, margin_width = 0):
    """ Imports the training files generated by augustus. This method is tied to the specific
     implementation of GeneMSA::getAllOEMsas and GeneMSA::getMsa.
     
    Args:
        paths (List[str]): Location of the file(s) generated by augustus (typically denoted *.out or *.out.gz)
        undersample_neg_by_factor (float): take any negative sample only with probability 1 / undersample_neg_by_factor,
                                           set to 1.0 to use all negative examples
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
    species = [leaf_order(c) for c in reference_clades] if reference_clades != None else []
    print ("species=", species, "reference_clades=", reference_clades)
    # total number of bytes to be read
    total_bytes = sum([os.path.getsize(x) for x in paths])

    # Status bar for the reading process
    pbar = tqdm(total = total_bytes, desc = "Parsing AUGUSTUS file(s)", unit = 'b', unit_scale = True)

    for p in range(len(paths)):
        path = paths[p]

        with (gzip.open(path, 'rt') if path.endswith('.gz') else open(path, 'r')) as f:

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
                                raise Exception(f'The species {list(encountered_species.values())} found in "{path}'
                                                f'" are not fully included in any of the given clades:' + str(C))
                            if num_parents > 1:
                                parent_clades = [reference_clades[i] for i in range(len(C)) if e <= C[i]]
                                raise Exception(f'The species {list(encountered_species.values())} found in "{path}" are included in multiple clades, namely in {parent_clades}.')
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
                    sidx = int(slice_data[0])
                    if sidx not in spec_reorder:
                        sys.exit(f"Error: species index {sidx} out of bounds in line\n{line}")
                    entry.spec_ids.append(spec_reorder[sidx])
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
        paths (List[str]): Location of the file(s) generated by PhyloCSF
        undersample_neg_by_factor (float): take any negative sample only with probability 1 / undersample_neg_by_factor, set to 1.0 to use all negative examples
        reference_clades (newick.Node): Root of a reference clade. The given order of species in this tree will be used in the input file(s). 
        margin_width (int): Width of flanking region around sequences
    
    Returns:
        List[MSA]: Training examples read from the file(s).
        List[List[str]]: Unique species configurations either encountered or given by reference.
    
    """

    # TODO: support either the non-clades case or make reference_clades a required parameter
    species = [leaf_order(c,use_alternatives=True) for c in reference_clades] if reference_clades != None else []

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


def plot_lenhist(msas, use_codons, id = "unfiltered"):
    """ plot the length distribution of classes (models) y=0, and y=1 to a pdf
    Returns:
        mlen  : array of alignment lengths
        labels: array of labels (= classes / model indices) 
    """
    num_alis = len(msas)
    mlen = np.zeros(num_alis, dtype = np.int32)
    labels = np.zeros(num_alis, dtype = np.int32)
    for i, msa in enumerate(msas):
        mlen[i] = msa.alilen(use_codons)
        labels[i] = msa.model
    
    fig, ax = plt.subplots(1, 2, figsize = (24, 8))
    colors = ["red", "green"]
    for label in [0, 1]:
        these_len = mlen[labels == label]
        ax[label].hist(these_len, bins = 200, range = [0, 2000], density = True, color = colors[label])
        ax[label].set_title("length distribution " + id + " \ny=" + str(label) + 
                           " n=" + str(np.sum(labels == label)) + " mean=" + str(np.round(np.mean(these_len), 1)))
    
    fig.savefig("lendist-oe-" + id + ".pdf", format = 'pdf')
    return [mlen, labels]

def subsample_lengths(msas, use_codons, max_sequence_length = 14999, min_sequence_length = 1):
    """ Subsample the [short] negatives so that
        the length distribution is very similar to that of the positives.
        Negative examples (model=0) of a length that is overrepresented compared to the
        frequency of that length in positive examples (model=1) are removed at random.
        Also, filter out 'alignments' with fewer than 2 sequences.
    Args:
        msas: an input list of MSAs
        use_codons: msas will be interpreted as a codon alignment
        max_sequence_length: upper bound on number of codons
        min_sequence_length: lower bound on number of codons
    Returns:
        filtered_msas: a subset of the input
    """
    ### compute and plot lengths and labels
    msas_in_range = []
    num_dropped_shallow = 0
    for msa in msas:
        if len(msa.sequences) < 2:
            num_dropped_shallow += 1
            continue
        length = msa.alilen(use_codons)
        if use_codons:
            length = int(length / 3)
        if (length >= min_sequence_length and length <= max_sequence_length):
            msas_in_range.append(msa)
    if (num_dropped_shallow > 0):
        print(f"{num_dropped_shallow} MSAs were dropped as they had fewer than 2 rows.")
    mlen, labels = plot_lenhist(msas_in_range, use_codons)
    assert (len(mlen) == len(labels) and len(mlen) == len(msas_in_range)), "length inconsistency"
    
    ### compute probabilities for subsampling
    max_subsample = 2001 # Don't apply subsampling for longer alignments, there typically are too few.
    distr = np.zeros([2, max_subsample], dtype = float)

    for i, slen in enumerate(mlen):
        if (slen < max_subsample):
            label = labels[i]
            distr[label, slen] += 1.0

    ratio = distr[1] / np.maximum(distr[0], 1.0) # overrepresentation ratio, for each length
    # as the sample has random variation, the ratios are smoothed before using them 
    ratio_smooth = np.zeros_like(ratio)

    def radius(slen):
        """ The offsets to the averaging interval, equal offsets leads to systematic overestimation
            Up to a length of 100 there is no smoothing. Beyond that, it is increasing.
        """
        return [int(.03 * max(slen - 100, 0)), int(.06 * max(slen - 100, 0))]

    for slen in range(1, max_subsample):
        r1, r2 = radius(slen)
        a = max(slen - r1, 0)
        b = min(slen + r2, max_subsample - 1)
        ratio_smooth[slen] = np.mean(ratio[a : b + 1])

    ratio_smooth /= np.max(ratio_smooth)

    fig, ax = plt.subplots(figsize = (6, 6))
    ax.plot(ratio_smooth, "b-")
    ax.set_title("length distribution subsampling probabilities")
    fig.savefig("subsampling-probs.pdf", format = 'pdf')

    filtered_msas = []
    for i, msa in enumerate(msas_in_range):
        slen = msa.alilen(use_codons)
        if msa.model != 0 or slen >= max_subsample or random.random() < ratio_smooth[slen]:
             filtered_msas.append(msa)

    plot_lenhist(filtered_msas, use_codons, id = "subsampled")
    print ("Subsampling based on lengths has reduced the number of alignents from",
           len(msas), "to", len(filtered_msas))
    return filtered_msas
    
def subsample_labels(msas, ratio):
    """ Subsample excess examples to that the ratio of negative to positive examples is 'ratio'
    Args:
        msas: an input list of MSAs
        ratio: fraction negatives/positives in the output
    Returns:
        filtered_msas: a subset of the input
    """
    print ("subsampling with ratio", ratio)
    
    if (ratio <= 0.0):
        print ("Warning: ratio_neg_to_pos must be positive. Skipping the subsampling based on label.")
        return msas
    
    num_neg = num_pos = 0
    pos_msas = []
    neg_msas = []
    filtered_msas = []
    for msa in msas:
        if msa.model == 0:
            num_neg += 1
            neg_msas.append(msa)
        elif msa.model == 1:
            num_pos += 1
            pos_msas.append(msa)

    if (num_neg > num_pos * ratio): # too many negatives
        reduced_size = int(num_pos * ratio + 0.5)
        print ("Reducing number of negatives from", num_neg, "to", reduced_size, ".")
        filtered_msas.extend(random.sample(neg_msas, reduced_size))
        filtered_msas.extend(pos_msas)
    else: # too many positives
        reduced_size = int(num_neg / ratio + 0.5)
        print ("Warning: --ratio_neg_to_pos removed positive examples. Maybe you want to omit the parameter to save positves.")
        print ("Reducing number of positives from", num_pos, "to", reduced_size)
        filtered_msas.extend(random.sample(pos_msas, reduced_size))
        filtered_msas.extend(neg_msas)

    random.shuffle(filtered_msas)
    return filtered_msas


def export_nexus(msas, species, nex_fname, n, use_codons):
    """ A sample of positive alignments are concatenated and converted to a NEXUS format that can be used directly by MrBayes to create a tree.
    Args:
        msas: an input list of MSAs
        nex_fname: output file name
        n: maximal sample size
    """
    positiveMSAs = []
    num_in_frame_stops = num_positives = 0
    for msa in msas:
        if (msa.model == 1):
            num_positives += 1
            msa.codon_aligned_sequences
            if msa.in_frame_stop:
                num_in_frame_stops += 1
            else:
                positiveMSAs.append(msa)
    num_pos = len(positiveMSAs)
    if n > num_pos:
        print ("Warning: Requested NEXUS sample size larger than the number of positive alignments (",
              num_pos, "). Taking all of them as sample.")
        n = num_pos
    if num_in_frame_stops > 0:
        print("Found", num_in_frame_stops, "in-frame stop codons in ", num_positives, "positive alignments. Omitting them.")

    sampledMSAs = random.sample(positiveMSAs, n)
    num_col = 0
    snameset = set()
    for msa in sampledMSAs:
        ca = msa.codon_aligned_sequences
        num_col += len(ca[0])
        # num_col += msa.alilen()
        for (c, l) in msa.spec_ids:
            snameset.add(l)
    sidxs = list(snameset) # all species names occurring in any chosen MSA
    
    # write as .nex file
    clade_specieslist = species[0]
    max_len_speciesname = max(len(name) for name in clade_specieslist)
    nexF = open(nex_fname, "w")
    nexF.write("#NEXUS\n")
    
    # data block (MSA)
    nexF.write("begin data;\n")
    nexF.write("dimensions ntax=" + str(len(sidxs))
               + " nchar=" + str(num_col) + ";\n")
    nexF.write("format datatype=dna interleave=yes gap=-;\nmatrix\n")
    for msa in sampledMSAs:
        msanames = [l for (c, l) in msa.spec_ids]
        
        ca = msa.codon_aligned_sequences
        codonalilen = len(ca[0])
        for sidx in sidxs: 
            nexF.write('{1:{0}}'.format(max_len_speciesname + 2, clade_specieslist[sidx]))
            # write alignment row
            try:
                i = msanames.index(sidx)
                nexF.write(ca[i])
            except ValueError: # row does not exist in this MSA, pad with gaps
                nexF.write("-" * codonalilen)
            nexF.write("\n")
        nexF.write("\n")
    nexF.write(";\nend;\n")
    
    # MrBayes command block
    nexF.write("begin mrbayes;\n")
    nexF.write("set autoclose=yes nowarn=yes;\n")
    #nexF.write("execute " + nex_fname + ";\n")
    nexF.write("lset nst=6 Nucmodel=Codon omegavar=M3 rates=gamma;\n") # for codon models: Nucmodel=Codon omegavar=M3
    nexF.write("mcmc nruns=1 ngen=100000;\n")
    nexF.write("sumt relburnin=yes burninfrac=0.2;\n")
    nexF.write("end;\n")

    nexF.close()
    
# TODO: Delete this function when debug is done. It is now directly implemented in the persistence function
def write_msa(msa, species, tfwriter, use_codons=True, verbose=False):
    """Write a coded MSA (either as sequence of nucleotides or codons) as an entry into a  Tensorflow-Records file.
    
    Args:
        msa (MSA): Sequence that is to be persisted
        use_codons (bool): Whether one should write onehot encoded sequences of codons which are codon-aligned 
                           or a onehot encoded sequences of nucleotides
        tfwriter (TFRecordWriter): Target to which the example shall be written.
        verbose (bool): Whether debug messages shall be written
    """
    

    # Use the correct onehot encded sequences
    coded_sequences = msa.coded_codon_aligned_sequences if use_codons else msa.coded_sequences
    
    # Infer the length of the sequences
    sequence_length = len(coded_sequences[1])  

    # cardinality of the alphabet that has been onehot-encoded
    s = coded_sequences.shape[-1]
    
    # get the id of the used clade and leaves inside this clade
    clade_id = msa.spec_ids[0][0]
    num_species = len(species[clade_id])
    leaf_ids = [l for (c,l) in msa.spec_ids]
    
    
    # embed the coded sequences into a full MSA for the whole leaf-set of the given clade
    S = np.ones((num_species, sequence_length, s), dtype = np.int32)
    S[leaf_ids,...] = coded_sequences
    
    # make the shape conform with the usual way datasets are structured,
    # namely the columns of the MSA are the examples and should therefore
    # be the first axis
    S = np.transpose(S, (1,0,2))
    


    # use model (`0` or `1`), the id of the clade and length of the sequences
    # as context features
    msa_context = tf.train.Features(feature = {
        'model': tf.train.Feature(int64_list = tf.train.Int64List(value = [msa.model])),
        'clade_id': tf.train.Feature(int64_list = tf.train.Int64List(value = [clade_id])),
        'sequence_length': tf.train.Feature(int64_list = tf.train.Int64List(value = [sequence_length])),
    })

    ## save `S` as a one element byte-sequence in feature_lists
    sequence_feature = [tf.train.Feature(bytes_list = tf.train.BytesList(value = [S.tostring()]))]
    msa_feature_lists = tf.train.FeatureLists(feature_list = {
        'sequence_onehot': tf.train.FeatureList(feature = sequence_feature)
    })

    
    # create the SequenceExample
    msa_sequence_example = tf.train.SequenceExample(
        context = msa_context,
        feature_lists = msa_feature_lists
    )

    # write the serialized example to the TFWriter
    msa_serialized = msa_sequence_example.SerializeToString()
    tfwriter.write(msa_serialized)

    
    
def preprocess_export(dataset, species, splits = None, split_models = None,
                      use_codons = False, verbose = False):
    
    # Prepare for iteration
    splits = splits if splits != None else {None: 1.0}
    split_models = split_models if split_models != None else [None]

    # Check whether splits are valid
    split_values_are_numbers = all([ isinstance(x, numbers.Number) for x in list(splits.values()) ])

    if not split_values_are_numbers:
        raise ValueError("The values of the dict `splits` must be numbers. But it is given by {splits}")

    # Convert relative numbers to absolute numbers
    random.shuffle(dataset)
    n_data = len(dataset)
    requested_sizes = np.array(list(splits.values()))
    negs = np.nonzero(requested_sizes < 0) # maximally fill the negative sizes, e.g. -1
    
    to_total = lambda x: int(max(x, 0) * n_data) if isinstance(x, float) else max(x, 0)
    
    split_totals = np.array([to_total(x) for x in requested_sizes])
    n_wanted = sum(split_totals)
    if len(negs) > 0 and n_wanted < n_data:
        # divide the remaining examples between the ones with requested negative size
        each_gets = int((n_data - n_wanted) / len(negs[0]))
        if verbose:
            print("Subsets (splits) with requested negative size each get ", each_gets, "alignents.")
        split_totals[negs] = each_gets
        n_wanted = sum(split_totals) # = n_data
    print("Split totals:", split_totals)
    
    # rescale accordindly
    if n_wanted > n_data:
        split_totals = split_totals * (n_data / n_wanted)

    # The upper bound indices used to decide where to write the `i`-th entry
    split_bins = np.cumsum(split_totals)
    # print ("split_bins=", split_bins , "\nsplits=", splits, "\nsplit_models=", split_models)
    return splits, split_models, split_bins, n_wanted


def persist_as_tfrecord(dataset, out_dir, basename, species,
                        splits=None, split_models=None, split_bins=None, 
                        n_wanted=None, use_codons=False,
                        use_compression=True, verbose=False):
    # Importing Tensorflow takes a while. Therefore to not slow down the rest 
    # of the script it is only imported once used.
    print ("Writing to tfrecords...")
    import tensorflow as tf

    options = tf.io.TFRecordOptions(compression_type = 'GZIP') if use_compression else None


    # Generate target file name based on chosen split and model
    
    target_path = lambda split_name, model: \
        os.path.join(out_dir, # this appends a slash if the user did not
                     basename
                     + ('-' + split_name if split_name != None else '')
                     + ('-m' + str(model) if model != None else '')
                     + '.tfrecord'
                     + ('.gz' if use_compression else ''))
    
    with ExitStack() as stack:
    
        #file_indices = [(split_name, model, ) for ]
        tfwriters = [[
                stack.enter_context(tf.io.TFRecordWriter(target_path(sn,m), options = options))
            for m in split_models] for sn in splits]

        n_written = np.zeros([len(splits), len(split_models)])
        
        for i in tqdm(range(n_wanted), desc="Writing TensorFlow record", unit=" MSA"):

            msa = dataset[i]
            iconfigurations = msa.spec_ids
            leaf_configuration = msa.coded_sequences
            model = msa.model

            # retrieve the wanted tfwriter for this MSA
            s = np.digitize(i, split_bins)
            m = split_models.index(model) if model in split_models else 0 
            tfwriter = tfwriters[s][m]
            n_written[s][m] += 1

            #Write a coded MSA (either as sequence of nucleotides or codons) as an entry into a  Tensorflow-Records file.
            # in order to do so we need to setup the proper format for `tf.train.SequenceExample`

            # Use the correct onehot encded sequences
            coded_sequences = msa.coded_codon_aligned_sequences if use_codons else msa.coded_sequences

            # Infer the length of the sequences
            sequence_length = coded_sequences.shape[1]

            # cardinality of the alphabet that has been onehot-encoded
            s = coded_sequences.shape[-1]

            # get the id of the used clade and leaves inside this clade
            clade_id = msa.spec_ids[0][0]
            num_species = len(species[clade_id])
            leaf_ids = [l for (c,l) in msa.spec_ids]


            # embed the coded sequences into a full MSA for the whole leaf-set of the given clade
            S = np.ones((num_species, sequence_length, s), dtype = np.int32)
            S[leaf_ids,...] = coded_sequences

            # make the shape conform with the usual way datasets are structured,
            # namely the columns of the MSA are the examples and should therefore
            # be the first axis
            S = np.transpose(S, (1,0,2))

            # use model (`0` or `1`), the id of the clade and length of the sequences
            # as context features
            msa_context = tf.train.Features(feature = {
                'model': tf.train.Feature(int64_list = tf.train.Int64List(value = [msa.model])),
                'clade_id': tf.train.Feature(int64_list = tf.train.Int64List(value = [clade_id])),
                'sequence_length': tf.train.Feature(int64_list = tf.train.Int64List(value = [sequence_length])),
                })

            ## save `S` as a one element byte-sequence in feature_lists
            sequence_feature = [tf.train.Feature(bytes_list = tf.train.BytesList(value = [S.tostring()]))]
            msa_feature_lists = tf.train.FeatureLists(feature_list = {
                'sequence_onehot': tf.train.FeatureList(feature = sequence_feature)
                })


            # create the SequenceExample
            msa_sequence_example = tf.train.SequenceExample(
                    context = msa_context,
                    feature_lists = msa_feature_lists
                    )

            # write the serialized example to the TFWriter
            msa_serialized = msa_sequence_example.SerializeToString()
            tfwriter.write(msa_serialized)

            #if verbose:
            #    print(f"leaf_configuration[1,...]: {leaf_configuration.shape} {leaf_configuration[1,...]}")
            #    ichar_test = np.random.randint(0, sequence_length)
            #    print("ichar_test:", ichar_test)
            #    print(f"Sample of the conversion:")
            #    print(f"\tModel: {imodel}")
            #    print(f"\tSequence Length: {sequence_length}")
            #    print(f"\tConfiguration: {iconfigurations}")
            #    print(f"\tPosition {ichar_test} character of the sequence: {leaf_configuration[:, ichar_test]}")
    
    print ("number of tf records written [rows: split bin s, column: model/label m]:\n", n_written)


def get_end_offset(start_offset, seqlen):
    """ 
       get the largest position where a complete codon could start,
       end_offset - start_offset is a multiple of 3 so incomplete codons are truncated
    """
    end_offset = seqlen - 3 # -3 so, a full codon could end right at end_offset
    end_offset = start_offset + math.floor((end_offset - start_offset) / 3) * 3
    return end_offset

    
def write_phylocsf(dataset, out_dir, basename, species,
                   splits = None, split_models = None, split_bins = None, 
                   n_wanted = None, use_codons = False):
    """
       Each MSA is written into a single text file in a format required by PhyloCSF.
    """
    print ("Writing to PhyloCSF flat files...")
    classnames = ["controls", "exons"]
    subdir_size = 500
    phyloDEBUG = False
    
    splitnames = list(splits.keys()) # e.g. train, val1, val2, test
           
    refid = None # 3 (dm) is reference,
    class_counts = [0, 0] # how many examples of class y=0 and y=1 have been seen yet
    n_written = np.zeros([len(splits), len(split_models)])
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
            
    for i in tqdm(range(n_wanted), desc = "Writing PhyloCSF dataset", unit = " MSA"):
        s = np.digitize(i, split_bins)
        split_dir = os.path.join(out_dir, basename, splitnames[s])
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
        
        msa = dataset[i]
         # get the id of the used clade and leaves inside this clade
        clade_id = msa.spec_ids[0][0]
        phyloCSFspecies = species[clade_id] # correct?  
    
        frame = msa.frame
        y = msa.model
        assert y == 0 or y == 1, "PhyloCSF output expects binary class"
        
        class_dir = os.path.join(split_dir, classnames[y])
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)  

        subdir = os.path.join(class_dir, "{:03d}".format(int(class_counts[y] / subdir_size)))
        if not os.path.exists(subdir):
            os.makedirs(subdir)

        fname = os.path.join(subdir, "{:03d}.fa".format(class_counts[y]))
        fa = open(fname, "w+")
        
        class_counts[y] += 1
        # indices to species ids sorted so reference is first
        # not required by PhyloCSF
        
        if refid:
            sids = sorted(range(len(msa.spec_ids)), key = lambda k: ((msa.spec_ids[k])[1] != refid))
        else:
            sids = range(len(msa.spec_ids))

        for j, k in enumerate(sids):
            (_, k) = msa.spec_ids[k]
            fa.write(">" + phyloCSFspecies[k])
            if j == 0:
                fa.write("|y=" + str(y) + "|f=" + str(frame))
                fa.write(" " + msa.chromosome_id + ":" + str(msa.start_index) + "-" + str(msa.end_index))
                if phyloDEBUG:     
                    fa.write((" +" if msa.is_on_plus_strand else " -") + " phase="  + str(frame))
            fa.write("\n")
            seq = msa.sequences[sids[j]]
            start_offset = frame
            end_offset = get_end_offset(start_offset, len(seq))
            
            if phyloDEBUG:
                fa.write(seq[0 : margin_width] + " " +
                      seq[margin_width: start_offset] + " ")
            
            fa.write(seq[start_offset : end_offset + 3])
            if phyloDEBUG:
                  fa.write(" " + seq[end_offset + 3 : len(seq)]
                      + " " + seq[len(seq):])
            fa.write("\n")
        n_written[s][y] += 1
        fa.close()
    print ("number of PhyloCSF records written [rows: split bin s, column: model/label m]:\n", n_written)
       

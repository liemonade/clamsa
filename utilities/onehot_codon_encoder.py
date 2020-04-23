import json
import itertools
import numpy as np

class OnehotCodonEncoderSingleton(object):
    def __init__(self):
        self._encoders = {}

    def _encoder(self, alphabet, codon_length, bucket_alphabet, use_bucket_alphabet=True):
        
        # create a dict. with the parameters
        encoder_config = {
            "alphabet": alphabet,
            "codon_length": codon_length,
            "bucket_alphabet": bucket_alphabet,
            "use_bucket_alphabet": use_bucket_alphabet
        }
        
        # dump into a string to make hashable
        encoder_key = json.dumps(encoder_config)
        
        # if the encocer has already been calculated we can retrieve it
        if encoder_key in self._encoders:
            return self._encoders[encoder_key]
        
        
        # calculate the words of length `codon_length` in the given alphabet
        # these will automatically be lexikographically sorted
        codons = itertools.product(*[alphabet for i in range(codon_length)])
        codons = [''.join(c) for c in codons]
        C = {c:i for (i,c) in enumerate(codons)}

        # calculate the 'buckets', i.e. all characters like 'n' in the fasta notation
        # which could be any of the characters 'acgt'
        all_codons = codons
        B = {c: c for c in alphabet}

        if use_bucket_alphabet:
            bucket_keys = ''.join(bucket_alphabet.keys())
            all_codons = itertools.product(*[alphabet+bucket_keys for i in range(codon_length)])
            all_codons = [''.join(c) for c in all_codons]
            B.update(bucket_alphabet)


        # use these buckets to calculate the dict `E`.
        # the keys are elements of `all_codons` and their values are
        # their onehot encoding
        E = {}

        for gen_codon in all_codons:

            possible_codons = itertools.product(*[B[c] for c in gen_codon])
            possible_codons = [''.join(c) for c in possible_codons]

            encoding = np.zeros(len(codons))
            encoding[[C[c] for c in possible_codons]] = 1
            E[gen_codon] = encoding
            
        # save the encoder for later reuse
        self._encoders[encoder_key] = E
            
        return E
    
    def encode(self,
               sequences, 
               alphabet="acgt",
               codon_length=3, 
               bucket_alphabet = {
                    'r': 'ag',
                    'y': 'ct',
                    'k': 'gt',
                    'm': 'ac',
                    's': 'cg',
                    'w': 'at',
                    'b': 'cgt',
                    'd': 'agt',
                    'h': 'act',
                    'v': 'acg',
                    'n': 'acgt'
                }, 
               use_bucket_alphabet=True):
        
        E = self._encoder(alphabet, codon_length, bucket_alphabet, use_bucket_alphabet)
        coder = lambda c: E.get(c, np.ones(len(alphabet) ** codon_length))
    
        coded_sequences = np.array([[coder(s[i:i+codon_length]) for i in range(0,len(s), codon_length)] for s in sequences])
        return coded_sequences
        

# Singleton Object
# In this way we can reuse the calculated encoders and do not
# need to recalculate them in each new `encode` call
OnehotCodonEncoder = OnehotCodonEncoderSingleton()
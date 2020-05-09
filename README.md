# aladdin
Aladdin (working title). Now with the epic quest of searching coding sequences in MSA's instead of a lamp in the Cave of Wonders...

# Needed packages
Assuming a working conda environment, the following commands will install all needed packages for aladdin
```console
conda install numpy tensorflow-gpu regex biopyton
conda install -c bioconda python-newick
conda install -c conda-forge tqdm

```

# Example Conversion
Example conversion parameters for our usual flie dataset from augustus:
```console
./aladdin.py convert augustus msa/train.out --clades clades/flies.nwk --splits '{"train": 0.9, "test": 0.05, "val": 0.05}' --margin_width 10 --basename augustus_flies --use_codons --split_models 0 1
```
Where we assume that the file `train.out` is stored in the folder `msa` and the Newick file specifying the phylogenetic tree is stored in `clades/flies.nwk`.

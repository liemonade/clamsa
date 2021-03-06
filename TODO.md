# Computational tasks
 - [x] 1) Generate training and test sets for human, fly.
 - [x] 2) Reproduce the fly results from the talk in January with the new code.
 - [x] 3) Check accuracy on human when trained on human and compare to fly (training & prediction).
 - [x] 4) Compute cross-training accuracy human version on fly test set and fly version on human test set.  
 - [x] 5) Train wth a mixed human and fly training set and evaluate on human and fly test set.
 - [x] 6) Conditional on relative success: Compile set of labeled non-animal alignments, tree and training and test sets.
 - [x] 7) Optional: compare to a second tool, PAML that computes omega=dN/dS
  
# Programming tasks
  - [x] a) Implement ```--to_phylocsf``` option.
  - [x] b) Implement `clamsa train`.
  - [x] c) Implement model specific training callbacks (drawing rate matrix progression for example)
  - [x] d) Implement `recover_model` function to load a model from its `Trial ID`
  - [x] e) Implement `clamsa predict`:
    - Present statistics on the datasets
    - Evaluate learned models on specified datasets
  - [ ] f) ClaMSA runs with the same input and similar output as PhyloCSF (**drop-in replacement**).
        This also enables the uniform evaluation of all competitors and is a check that the very same
        data is used for testing and the very same scripts for evaluation.

# Other Tasks
  - [x] Find name soon so figures could be final, e.g.
     - **ClaMSA**
     - **DiscrEvo**: discriminative evolutionary model
     - **ELEMO**: *E*nd-to-end *L*earning of *E*volutionary *MO*dels 
     - some acronym from: differentiable rate matrix pruning model discriminative ML evolution tree Markov selection
  - [x] obtain **biological results**  
        With modest effort, we could screen the negatively labeled MSAs that are most confidently predicted as positive for candidates of missing genes.
        Finding an unannotated human gene is a long shot, but I could search genome-wide with a non-human vertebrate as reference.
        This could lead to a figure of one - hopefully clear - MSA example of a yet missing gene and some statistics to estimate missing protein-coding genes.
        This may be a soft requirement for submitting to high-ranking journals as "Bioinformatics" (Oxford, impact factor 5.6).

# Computational tasks
 - [x] 1) Generate training and test sets for human, fly.
 - [x] 2) Reproduce the fly results from the talk in January with the new code.
 - [ ] 3) Check accuracy on human when trained on human and compare to fly (training & prediction).
 - [ ] 4) Compute cross-training accuracy human version on fly test set and fly version on human test set.  
 - [ ] 5) Train wth a mixed human and fly training set and evaluate on human and fly test set.
 - [ ] 6) Conditaional on relative success: Compile set of labeled plant alignments, tree and training and test sets.
 - [ ] 7) Optional: compare to a second tool, PAML that computes omega=dN/dS
  
# Programming tasks
  - [x] a) Implement ```--to_phylocsf``` option.
  - [x] b) Implement `aladdin train`.
  - [ ] c) Implement model specific training callbacks (drawing rate matrix progression for example)
  - [ ] d) Implement `recover_model` function to load a model from its `Trial ID`
  - [ ] e) Implement `aladdin evaluate`:
    - Present statistics on the datasets
    - Evaluate learned models on specified datasets
  
# Notes
 

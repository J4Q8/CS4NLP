# Memory-based generation: Enabling LMs to handle long input texts by extracting useful information

As part of the course "Computational Semantics for NLP" at ETH Zurich, we are exploring different methods of enabling transformer-based LMs to handle 
long text sequences. Our baseline methods include Random selection, Bias towards Start+End, TF-IDF. Then, we explore the use of sentence embeddings with SBERT and experiment with different chunk sizes. Our experiment uses different LMs to perform question answering on the QuALITY dataset in order to find the most fitting one for that task.


## Reproduce the experiments
To perform the experiments for the different models, run the Notebooks:
- Experiments_Deberta.ipynb
- Experiments_Longformer.ipynb
- Experiments_Roberta.ipynb

Make sure that all .py files are included in the project.

## Reproduce the analysis
To reproduce the plots used in our analysis section, run the Notebooks:
- Analysis.ipynb

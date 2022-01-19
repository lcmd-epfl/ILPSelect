# Data
## Structures
The matrices for 3 target structures (to synthesize) and a database of 7165 query structures (to combine to build the target)
are compressed in `data.npz` 

Within python, it can be read like: 
```
data = np.load("data.npz", allow_pickle=True)
```

where `data.files` will return the names of the numpy arrays (should be `target_labels, target_CMs, target_ncharges, database_labels, database_CMs, database_ncharges`) 
where CMs are the matrices (of target and database respectively) and the corresponding arrays can be accessed like: 

```
data["target_labels"]
```

For more details see the documentation: 
https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.savez.html

## Connectivity / functional group information
Adjacency matrices and functional group information derived from the connectivity are compressed in `connectivity_data.npz`. 

Within python, it can be read like:
```
connectivity_data = np.load("connectivity_data.npz")
```

the corresponding keys are `fg_counts_targets` for the functional group counts of each of the 3 target molecules,`fg_counts_frags` for the functional group counts of
each of the fragment molecules, `frag_adj_matrices` for the adjacency matrices of the fragments and `target_adj_matrices` for the adjacency matrices of the target molecules.
The order is the same as those in `data` containing the structures.  

## Optimal databases 
Dedicated databases of small molecules are saved for each target, all compressed in the file `amons_data.npz`.

```
data = np.load("amons_data.npz")
```

contains the same information as in the original `data`, but now specific to each target. 
Target 0 (qm9) has the data: 
```
qm9_amons_labels
qm9_amons_ncharges
qm9_amons_CMs
```
where the CMs are the representation matrices. 

Similarly, target 1 (vitc) has the same data with the prefix `vitc_`. Same for vitd. These databases are much smaller, making the search faster.

### Optimal databases and vector data 
Rather than using symmetric matrices to represent our molecules where each row/column index represents an atom index, we can directly use a vector of the same length for each atom index. 
In other words, we have an asymmetric matrix of dimensions N_atoms x V_dim where V_dim is the length of the vector. We can access the representation for each atom as the appropriate index
of the asymmetric matrix. V_dim will vary based on the atoms present in the target system, but will be consistent between the target and database candidates.

Now we have datasets for 4 different asymmetric representations: aCM, SLATM, SOAP and FCHL, all named like `target_repname_data.npz` for the target and `amons_repname_data.npz` for the fragments.

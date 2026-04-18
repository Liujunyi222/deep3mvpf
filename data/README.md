# Data files

## Included regression data

- `GSE106677_sequences.fasta`: sequence file used by the RNA degradation regression task.
- `output_structure.txt`: RNA secondary structure file aligned by sequence ID.
- `degradation.xlsx`: degradation targets used for regression.

## m6A dataset layout

If you also include the m6A dataset in this repository, the recommended location is:

```text
data/m6a_final_data_3utr/
  train/
    A549/
      A549_train_data_with_struct.tsv
      A549_train_fold0.fa
      ...
      A549_train_fold9.fa
    CD8T/
    ESC/
    HCT116/
    HEK293/
    HEK293T/
    Hela/
    HepG2/
    MOLM13/
  validation/
  test/
```

For each split (`train`, `validation`, `test`) and each cell line, keep:

- one `*_data_with_struct.tsv` file
- ten fold FASTA files: `*_fold0.fa` to `*_fold9.fa`

Example:

```text
data/m6a_final_data_3utr/test/A549/
├── A549_test_data_with_struct.tsv
├── A549_test_fold0.fa
├── ...
└── A549_test_fold9.fa
```

The scripts also support a custom location through `--root_path`.

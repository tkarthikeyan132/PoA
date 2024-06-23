# PoA
This is the official repository of "Eulerian at BioLaySumm: Preprocessing Over Abstract is All You Need" [ACCEPTED IN BioNLP Workshop at ACL 2024]

## Training the Model

To train the model, use the following command:

```bash
bash run_model.sh train <path_to_data> <path_to_save>
```

where *path_to_data* is a directory containing all the train and validation JSONL files for both
eLife and PLOS datasets. *path_to_save* is also a directory where the trained model must
be saved.

## Testing the Model

To test the model, use the following command:

```bash
bash run_model.sh test <path_to_data> <path_to_model> <path_to_result>
```

*path_to_data* is a directory containing the two test files, *path_to_model* is a directory containing the model stored by the training command. Finally, *path_to_result* is a directory
where the predicted summary files plos.txt and elife.txt must be stored.
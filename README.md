# fontRecognition-22928
A program for recognizing fonts in text appearing in everyday photos - my final project in a Computer Vision course I took in Fall 2022 in The Open University of Israel.

The training data for this project can be found [here](https://drive.google.com/drive/folders/1jzHYpTwywUYA53nMGHVROSuVO14hEueq?usp=sharing). Since I've opened a github account a few months after the semester was already over, I couldn't find the unlabeled test set to upload here.

To produce an output, run `classifying.py`. This program processes a dataset stored in `SynthText_test.h5`, and produces an output file `results.csv` containing information about the predicted font of every character appearing in it.

To visualize the model's performance on the test partition of the dataset (produced by `buildup.py`), run experimenting.py.

Either way, a full detailed description about what each program does exactly can be found in the comments at the top of each file. You are also free to read my submitted project report, to thoroughly understand my algorithm.

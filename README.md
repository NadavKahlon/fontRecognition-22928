# fontRecognition-22928
A program for recognizing fonts in text appearing in everyday photos - my final project in a Computer Vision course I took in Fall 2022 in The Open University of Israel.

The training data for this project can be found [here](https://drive.google.com/drive/folders/1jzHYpTwywUYA53nMGHVROSuVO14hEueq?usp=sharing). Since I've opened a github account a few months after the semester was already over, I couldn't find the unlabeled test set to upload here.

The project consists of 4 files:
> `buildup.py` - for preprocessing the data and splitting it to train, test, and validation partitions. The output of it for our data can be found [here](https://drive.google.com/file/d/1xXfunmTc3USouiQ1MdGVLN6XDUCeNtrz/view?usp=sharing).\
> `training.py` - for training the model(s).\
> `classifying.py` - to produce predictions for an unlabeled test set.\
> `experimenting.py` - to visualize the system's performance on a labeled set, using various metrics and plots.

A full and (very) detailed description about what each program exactly does can be found in the comments at the top of each python file. Feel free to read my submitted project report, to thoroughly understand the algorithm and the model I designed, alongside various experiments I performed to test it.

# soloTranscription

## Details
### configs/
This directory contains all the config files managed by hydra.

### data/
This directory is for storing the electric guitar tab, audio and preprocessed data of the datasets.

### src/
This directory contains the source codes.


## Setup
Download audio with following codes:
```

```

## Running the code
For note attribute prediction
```
cd note_attrib_prediction
python src/train.py
```
This will generate predicted note attribute to tech_prediction.
Then for full tab prediction
```
cd ..
cd tech_prediction
python src/train.py
```



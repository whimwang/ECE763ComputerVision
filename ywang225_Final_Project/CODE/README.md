# README: how to reproduce

## Steps
1. download the datasets from dropbox and unzip
2. install two packages
3. run the notebook you are interested in

## Datasets:

1. all datasets are stored in **the dropbox link: "https://www.dropbox.com/sh/f7zymuqmfqt4n64/AADuJc9A2t2_Ah9hTs9HXozpa?dl=0"**
	1. train.zip contains those images
	2. train.csv contains the relationship between image names and their labels
	3. h2p_table.csv, p2htable.csv and bounding-box.pickle are three intermediate results to help us pre-process the data
	4. CNN_NN.model is the trained model in our report to help us produce the visualization results. 
2. **change the path** in the second block in each notebook to the root path of the dataset folder on your computer
3. **mannuly unzip train.zop**


## Package:
1. run the **"pip" code in the first block to install the required two packages(ImageHash, keras-tqdm)**
2. the other  common packages are assumed to be already installed.

## Notebook contents:
1. For more clear presentation, we use python notebook form to present the results.
1. Each notebook corresponds to a result in our results section. The name is the same with that in the table.
2. The visualization.ipynb is for the result visualization part.
2. **We only need to run the whole notebook to see the corresponding results.**

## Claim:
1. It is our first time to write such a self-contained report, so if there were any problems, please feel free to send an email to us and we are more than happy to help explain it.

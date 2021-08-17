# Gaze Track
<img src="eye.jpg" alt="Gaze Track" />

Welcome to the complete guide for the implementation and experiments based on Google's recent paper [Accelerating eye movement research via accurate and affordable smartphone eye tracking](https://www.nature.com/articles/s41467-020-18360-5). 

Please use this index to quickly jump to the portions that interest you most.
- [Gaze Track](#gaze-track)
  * [Introduction](#introduction)
  * [The Dataset](#the-dataset)
  * [The Network](#the-network)
  * [Training](#training)
  * [Results](#results)
  * [Experiments](#experiments)

***

## Introduction
Eye tracking has many applications from driver safety to improved accessibility for people with disabilities. Current state of the art eye trackers are very expensive and tend to be bulky systems that need to be carefully setup and calibrated. The pervasiveness of handheld devices with powerful cameras have now made it possible to have high quality eye tracking right in our pockets!

The paper implemented here reports an error of 0.6–1° at a viewing distance of 25–40cm for a smartphone. This means if you look at a spot on the phone from a distance of 25–40cm, the algorithm can predict the location of the spot within an error of 0.46±0.03cm.

The authors have not open sourced code or provided trained models. The aim of this project therefore is to replicate the results reported and then extend the functionality to also predict head position and more.


***

## The Dataset
All trained models provided in this project are trained on some subset of the massive [MIT GazeCapture dataset](https://gazecapture.csail.mit.edu/index.php) that was released in 2016. You can access the dataset by registering on the website. 

### Raw Dataset Numbers
The figure below shows the number of participants per device as well as the train/val/test split as provided by the GazeCapture team. 

<img src="usersVSdevices.png"/>

Details of the file structure within the dataset and what information is contained are explained very well at the [Official GazeCapture git repo](https://github.com/CSAILVision/GazeCapture). 

For training the network and the different experiments, we split this large dataset based on a variety of filters and train/test/val combinations. These splits and how to generate them using the code are briefly described below. 

### Key Point Generation
Since the Google Model requires eye landmark key points that are not included in the GazeCapture dataset, converting from GazeCapture to a dataset usable for this project is a two step process. 
1. Extract gazecapture.tar to a temp folder
2. Extract *.tar.gz into the same temp folder
3. Use one of the dataset conversion scripts in `Utils/dataset_converter*` to change the folder structure to a usable one 
4. Use the [Utils/add_eye_kp.py](../Utils/add_eye_kp.py) file to generate the key points. 

### MIT Split - 
All frames that make it to the final dataset contains only those frames that have a valid face detection along with valid eye detections. If any one of the 3 detections are not present, the frame is discarded. 

The _MIT Split_ maintains the train test validation split at a per participant level, same as what GazeCapture does. What this means is that a data from one participant does not appear in more than one of the train/test/val sets. We have different participants in the train, val and test sets. This ensures that the trained model is truly robust and can generalize well.

You can use the [Utils/dataset_converter_mit_split.py](../Utils/dataset_converter_mit_split.py) file to generate the two datasets mentioned below.

#### Only Phone Only Portrait
The first dataset we will discuss is the closest to what Google used to train their model. We apply the following filters:
* Only phone data
* Only portrait orientation
* Valid face detections
* Valid eye detections 

This dataset is what the [provided base model](../Checkpoints/GoogleCheckpoint_1.ckpt) is trained on. 

The figure below shows the distribution of number of frames per device. 
<img src="MITSplitPort.png"/>

Overall, there were
* 501,735 Total frames from 1,241 participants
* 427,092 Train frames from 1,075  participants
* 19,102 Validation frames from 45 participants
* 55,541 Test frames from 121 participants

#### Only Phones All Orientations
The next dataset continues to split the data as suggested by GazeCapture but includes all the orientations. The following filters are applied: 
* Only phone data
* Valid face detections
* Valid eye detections

This dataset is used to train the model described in the `Experiments` folder. 

The figure below shows the distribution of number of frames per device. 
<img src="MITSplitAll.png"/>

Overall, there were
* 501,735 Total frames from 1,241 participants
* 427,092 Train frames from 1,075  participants
* 19,102 Validation frames from 45 participants
* 55,541 Test frames from 121 participants

### Google Split -
Google split their dataset according to the unique ground truth points. This therefore means that frames from each participant are present in the train test and validation sets. To ensure no data leaks though, frames related to a particular ground truth point do not appear in more than one set. The split is also a random 70/10/15 train/val/test split compared to a 13 point calibration split. 

You can use the [Utils/dataset_converter_google_split.py](../Utils/dataset_converter_google_split.py) file to generate this dataset.


### Test Split SVR 13 Point Calibration - 

### Test Split Google - 
***

## The Network

***

## Training

***

## Results

***

## Experiments

***

## References

*** 

## Acknowledgements
# GSoC 2021 Progress Tracking

## Week 1 (8<sup>th</sup> - 12<sup>th</sup> June 2021)

### What did you do this week?
* Created sub-dataset based on the paper
* Trained first model based on parameters provided in the paper
* Testing scripts to visualize and evaluate model performance. 
### What do you plan to do for next week?
* Model accuracy does not match what is reported. Finetuning of the model will be required over the coming weeks
* Test other existing implementations to check outputs. 
### Are you blocked on anything?
* No
***


## Week 2 (14<sup>th</sup> - 18<sup>th</sup> June 2021)

### What did you do this week?
* Trained models with a variety of tweaks - Different activation function, different batch size, inverted axis
* Questions for the authors of the paper
* Begin working on app for data collection
### What do you plan to do for next week?
* Complete data collection app and collect some data
* Fix the model using inputs from the authors and mentors
### Are you blocked on anything?
* No
***

## Week 3 (21<sup>st</sup> - 25<sup>th</sup> June 2021)

### What did you do this week?
* Figured out multi GPU training with PyTorch Lightning
* Trained MIT CSAIL model for better comparison
* Improved model accuracy
### What do you plan to do for next week?
* Complete data collection app and collect some data
* Complete testing framework for better understanding model performance
### Are you blocked on anything?
* Inputs from the author will hopefully come sometime next week. 
***

## Week 4 (28<sup>st</sup> June - 02<sup>nd</sup> July 2021)

### What did you do this week?
* Received inputs from authors and began incorporation of new methods
* Better results and testing framework created
* Setting up Neptune.ai for experiment tracking
* Began SVR coding for final step
### What do you plan to do for next week?
* Follow up questions for authors
* Additional documentation
* Further testing and fine tuning
### Are you blocked on anything?
* No
***

## Week 5 (05<sup>th</sup> - 09<sup>th</sup> July 2021)

### What did you do this week?
* SVR based personalization experiments started
* Trained device specific models using the pre trained larger model
* Moved to comet.ml for experiment tracking
* Documentation
### What do you plan to do for next week?
* Learn a simple affine transform instead of the SVR
* Finish the SVR based personalization
* Collect data using the app and check if it works
### Are you blocked on anything?
* No
***

## Week 6 (12<sup>th</sup> - 16<sup>th</sup> July 2021)
### What did you do this week?
* Completed SVR and Affine based fine tuning code
* Trained and tested device specific models using the pretrained larger model
* Started iTracker experimentation
### What do you plan to do for next week?
* Per person model fine tuning
* More iTracker experiments
* Compare results with the Google trained model once they provide it. 
### Are you blocked on anything?
* License agreement for the trained Google model
***

## Week 7 (19<sup>th</sup> - 23<sup>rd</sup> July 2021)
### What did you do this week?
* Fixed Affine transform and obtained reasonable results
* Base model finalized
* Fine tuning experiments - person based and device based
### What do you plan to do for next week?
* Self train iTracker model
* Experiment on the binary provided by Google
* Move towards final demo implementation
### Are you blocked on anything?
* no
***

## Week 8 (26<sup>th</sup> - 30<sup>th</sup> July 2021)
### What did you do this week?
* Started iTracker experiments
* Trained iTracker from scratch
* Fine tuned provided iTracker pyTorch model
* SVR fine tuning code fixed and working.

### What do you plan to do for next week?
* Experiment on the binary provided by Google
* Continue iTracker experiments
* Move towards final demo implementation
### Are you blocked on anything?
* Need to get the Google model file
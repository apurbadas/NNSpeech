## Last updated: 04/24/2016 for Spring 2016 semester project

## Change the value of 'ind' for selecting which feature (mfcc: 0, filterbank: 1, both: 2) to use in the feature vector for classification

-------------------------------------------------------------------

theano_classify.py :  This is the main file where I call theano_feature.py and tune the parametes for NN based classification.

theano_feature.py: NN classifier is implemented here for features extracted from timit database

svm_classify.py: SVM based classifier

features folder contains the feature extraction algorithms (mfcc, log filterbank)

-------------------------------------------------------------------



Currently, I am using speech features instead of direct samples of speech signal collected from timit database.


Features include: Log Filterbank Energies and Mel Frequency Cepstral Coefficients

feature extracted from each 25ms window speech signals in every 10ms of speech

80ms (16KHz) speech provides 7 frames of speech
filterbank: 26 coefficients
mfcc: 13 coefficients

Total features = (26+13)*7 = 273 for each sample of vowel phoneme

Total vowels = 12



--------------------------------------------------------------------


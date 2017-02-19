#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image0]: ./output_images/data_viz.png "Visualization"
[image1]: ./output_images/histo1.png  "Visualization"
[image2]: ./output_images/histo2.png  "Visualization"
[image3]: ./output_images/lenet5.png "Random Noise"
[image4]: ./traffic_signs/10sign.png "Traffic Sign 1"
[image5]: ./traffic_signs/blue_up.png "Traffic Sign 1"
[image6]: ./traffic_signs/sign2.png "Traffic Sign 1"
[image7]: ./traffic_signs/sign3.png "Traffic Sign 1"
[image8]: ./traffic_signs/stopsign.png "Traffic Sign 1"
[image9]: ./traffic_signs/unlimited.png "Traffic Sign 1"
[image10]: ./output_images/result10.png "Traffic Sign 1"
[image11]: ./output_images/resultexc.png "Traffic Sign 1"
[image12]: ./output_images/resultstop.png "Traffic Sign 1"
[image13]: ./output_images/resultunlimi.png "Traffic Sign 1"
[image14]: ./output_images/resultarrow.png "Traffic Sign 1"
[image15]: ./output_images/result80.png "Traffic Sign 1"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the first code cell of the IPython notebook.  

I used the numpy to calculate summary statistics of the traffic
signs data set:

* The size of training set is 65818
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the second and third code cell of the IPython notebook.  
Here is a sample of each class for the training data:

![alt text][image0]

 
The next image shows the distribution of samples among different classes

![alt text][image1]



###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

I applied random rotations to images from classes that have relatively few samples.
After doing this the distribution of samples became more homogeneous

![alt text][image2]


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. 


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the fifth cell of the ipython notebook. 

I used the LeNet architecture

![alt text][image3]


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the seventh cell of the ipython notebook. 

To train the model, I used an Adam optimizer starting with a learning rate of 0.0002
For the loss I used the softmax cross entropy, a batch size of 128 and 30 Epochs.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the eight cell of the Ipython notebook.

My final model results were:
* validation set accuracy of 0.969
* test set accuracy of  0.875

If a well known architecture was chosen:
I chose LeNet architecture because I noticed in the course that it was very powerful and reached high levels
of accuracy with very few epochs. This was confirmed with my tests and the model was successfully trained,
reaching over 90% validation accuracy and a testing accuracy not too far behind (85%).
###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]

Notice that the 10 sign was not available in the training data.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]

For each row of images, the left most image is the image that needs to be classified.
The classification result is shown on the top of each image row. The integer array displays
classified labels ordered by classification certainty. The float array shows the probability of each
label.

The model was able to correctly guess 4 of the 5 traffic signs it was supposed to know, 
which gives an accuracy of 80%. This is similar to the accuracy on the test set of 85%.
The mistake it made with the "80" sign is understandable. It confused the 3 to be an 8.

When giving the model an unknown image (here "10") one can see that it still was able to pick up all signs
that are very similar (round with red borders). Which means that the model has built good spacial and color
features in its layers.
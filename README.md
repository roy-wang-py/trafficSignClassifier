# **Traffic Sign Recognition** 

## Writeup


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

[image1]: ./examples/visualization_train.jpg "Visualization Train"
[image2]: ./examples/visualization_valid.jpg "Visualization Valid"
[image3]: ./examples/visualization_test.jpg "Visualization Test"
[image4]: ./examples/grayscale.jpg "Grayscaling"
[image5]: ./examples/processimg.jpg "Random Noise"
[image6]: ./examples/placeholder_1.png "Traffic Sign 1"
[image7]: ./examples/placeholder_2.png "Traffic Sign 2"
[image8]: ./examples/placeholder_3.png "Traffic Sign 3"
[image9]: ./examples/placeholder_4.png "Traffic Sign 4"
[image10]: ./examples/placeholder_5.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/roy-wang-py/trafficSignClassifier.git)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]
![alt text][image2]
![alt text][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it worked better in gray space according to the scientific paper Yann LeCun. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image4]

As a last step, I normalized the image data because it make 

I tried to generate additional data because images number per class are quite different.

To add more data to the the data set, I used the following techniques because these could  increase robustness.
1) rotation
2) translation
3) scaling

Here is an example of an original image and an augmented image:

![alt text][image5]

The difference between the original data set and the augmented data set is the following ... 

(Sadly, my model did worked well using these additional data.So my final model is trained using original data)

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x6 	|
| RELU					|												|
| Dropout               |                                               |
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 28x28x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 12x12x200	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x6x200				    |
| Concat                | 2 cov layers                                  |
| Dropout               |                                               |
| Fully connected		| output 120        							|
| RELU					|												|
| Dropout               |                                               |
| Fully connected		| output 84        							    |
| RELU					|												|
| Fully connected		| output 43        							    |
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?(i did not record it)
* validation set accuracy of 0.984 
* test set accuracy of 0.975

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I tried LeNet first, as we know it works well , reaching 0.93.

* What were some problems with the initial architecture?
According to the scientific paper Yann LeCun, concatting 2 cov layers might be a better way.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
Based on LeNet, i add another cov layer, and concatting 2 cov layers to fully connected layer. I try many ways, such as ConvNet, ConvNet2, ConvNet3, all based on LeNet, but add more layers. Finally ConvNet3 stands out.

* Which parameters were tuned? How were they adjusted and why?
I tried keep_prob, and found 0.5 works well. It could increase robustness, so works well with new images.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Convolution layer could help find relevance among pixels, so we can find some special shapes in images. Wihle dropout could help reduce amount of calculation, and avoid over fitting.

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10]

The Second image might be difficult to classify because the image contains part of other signs.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| General caution     	| General caution								|
| Ahead only			| Ahead only									|
| Speed limit (20km/h)	| Speed limit (20km/h)					 		|
| No vehicles			| No vehicles      							|


The model was able to correctly guess all 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 97.4%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 22th cell of the Ipython notebook.

For the Second image, the model is relatively sure that this is a General caution (probability of 0.929), and the image does contain a General caution. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .929         			| General caution   							| 
| .043     				| Traffic signals 								|
| .018					| Keep right									|
| .004	      			| Wild animals crossing			 				|
| .002				    | Speed limit (30km/h)     						|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?



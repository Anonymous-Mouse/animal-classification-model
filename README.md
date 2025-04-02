# animal-classification-model
CS105 Project using KNN and K-Means Clustering to automate animal identification of wildlife creatures

## Project Proposal
A topic that interests us is the classification of wildlife species from the [KTH-Animals dataset](https://www.csc.kth.se/~heydarma/Datasets.html). This topic serves as a practical introduction into the classification of species from images submitted by animal enthusiasts, hikers, or biologists. By investigating the features unique to some species, we learn more about how to employ machine learning methods to automate animal identification. With the techniques learned here, we hope to facilitate the detection of endangered animal species, improve public awareness, help conservation efforts, and guide future projects.

We want to know if it is possible to identify an animal by testing a number of features such as ear shape and size, snout size and shape, fur color, or tail size. We can answer this question by choosing all of our target classes. Then, we train some classification algorithms to recognize features unique to some target classes and cluster images with high ingroup similarity and low outgroup similarity.

## Data Collection and Cleaning
Using the KTH-Animals dataset, we extracted features from labeled images of leopards, coyotes, and zebras. To extract the data, we used HOG(Histogram of Oriented Gradients), mean RGB values, and edge values. The features obtained from our feature selection algorithm become quantitative variables about the characteristics of our animals.

The data is already collected, but we need to limit our dataset size for this project to 100 images. We also need to resize images to the same standard and normalize them before performing any analysis.

## Learning Models
We used K Nearest Neighbors to classify our images based on our features and K-means clustering to find natural groupings of data based on their feature values. By using clusters to classify images, we eliminated the need to test multiple values of K which helped in choosing K for KNN. We evaluated the success of our classification algorithm using a confusion matrix focusing on precision, accuracy, and recall.

## Contributions
Adolfo Plata - Introduction, Proposal, Hypotheses, KNN models , Feature extractions for Edges, avgRGB, and HOG functions. K-Mean clusters for Edges and HOG. Also, confusion matrices.

William Kim - RGB intensity models for KNN, feature extraction method for RGB flattening, project report descriptions for k-neighbors. Conclusion, confusion graph explanations, report descriptions

Iliyan Alibhai - KNN modelling for HOG, Slides and Presentation, report Descriptions,

Jalen Dioneda - K-Mean Clusters for both Edges, HOG, and RGB Intensity values. Descriptions and explanations for clusters,

Jarnett Asuncion - Slides and Presentation, Report descriptions
## Feature Selection
### Feature 1: Average RGB Value
To classify images, we first need to extract certain features from the images.
The first feature extraction we picked is grabbing the average RGB values from each image. This simple extraction grabs a single value to represent an image. We start here because it is the most basic and because our animals we classify live in wildly different environments (white snow vs green pastures), this may be a feature to select.

Examples of Average RGB values for each image in the dataset:
 89.8394318664966,
 94.44223666028911,
 98.98301977040816,
 61.9900284332483,
 159.65823634141157,
 44.38875823767007,
 71.32028592687075,
 64.98414248511905,
 57.32059816113946,
 113.85174186862245,
 97.3540271577381,
 84.68575946003402,
 102.97741948341837,
 71.90921290391157,
 110.79761904761905,
 86.64779310161565,
 84.66685932185374,
 79.91911803784014,
 107.26071561437075,
 134.90375876913265,
 130.39055856717687,
 138.5334555697279,
 96.2911684204932,
 104.4295347045068,
 158.59460698341837,
 106.35234640731292,
 83.26395089285714,
 116.08258264243197,
 58.38272613732993,
 120.79464950042517,
 72.11506829294218,
 96.71882971938776,
 80.1015625,
 73.68033854166667,
 59.86532073767007,
 106.00562686011905,
 144.52980176445578,
 115.76766448767007,
 83.26669456845238,
 137.1366656037415,
 86.41972921981292,
 73.51492081207483,
 71.23236872874149,
 57.55005713222789,
 62.22006536989796,
 115.63165656887755,
 87.21411298894557,
 58.254570578231295,
 27.12331260629252,
 157.10481770833334,
 77.75315555909864,
 75.31694435586735,
 87.34470663265306,
 97.36844972363946,
 53.28451849489796,
 101.72160661139456,
 74.85936835671768,
 106.9419244260204,
 84.23250159438776,
 114.80301339285714,
 121.5493928039966,
 112.05988919005102,
 80.17703018707483,
 79.15692761479592,
 107.01283482142857,
 71.24591438137755,
 100.52583572491497,
 78.86595849277211,
 77.00538770195578,
 67.98148517219387,
 71.03150244472789,
 172.1312181122449,
 68.97566565688776,
 102.27198926445578,
 126.96069169855443,
 85.51439599277211,
 162.8460685055272,
 175.96322278911563,
 130.48667357568027,
 132.71997236394557,
 79.17849170918367,
 106.46146896258503,
 118.13324431335035,
 122.12445525085035,
 48.15613706420068,
 66.47331393494898,
 133.8825069090136,
 90.74888392857143,
 87.04571906887755,
 145.62778353528913,
 68.8913956207483,
 55.32642431972789,
 106.26388446003402,
 113.95038132440476,
 62.62953736181973,
 79.16849356930273,
 56.52792835884354,
 91.55478050595238,
 112.30147879464286,
 65.8598931760204

### Feature 2: Histogram of oriented gradients (HOG)
Next, we want to grab the HOG of all the images as the next feature descriptor. HOG is a well-known feature descriptor for image classification, specifically object identification. We want to try this out.
![Histogram of Oriented Gradients for Zebra]()
Using 'zebras' as an example, we see the output of a 2D array (A list of 1D arrays describing an image), and a HOG graph to depict the gradients of the images.

### Feature 3: Edges
The second feature we decide to select for are the edges of a picture. This extractEdgesFunction transforms each image and only shows the outlines that the CV2 library identifies.

You can see the new images below.
![Edge Feature of Zebra]()
As you can see, all the images have been converted so that the outlines become the main feature.

## More data processing : resizing resolution
Next, we also decided to compute the avg dimensions of each image. The point of this function was for preprocessing purposes. Every image has to be resized to be the same resolution, as we will be using a 1D array dimensional space to compare images and classify them. In order to do that, the size of the dimensions must be the same in order to properly compare.

Zebra image Dimensions: 
Average Height: 281
Average Width: 342
Leopard image Dimensions: 
Average Height: 276
Average Width: 364
Coyote image Dimensions: 
Average Height: 294
Average Width: 346

We see the average resolutions of each category. We decide on 224 by 224.

## FEATURE SELECTION
Now that we have our features, we want to select some features to build our classification models. We decide to try out HOG, average RGB, edge outlined images, and RGB values. We will explain indepth as we come across each subject.

For now, we will focus on building our models.

We will first focus on our supervised learning models : KNN.
## K-Nearest Neighbor
Model 1 : Using HOG
For our first model, we use HOG as it's main feature.
We first convert all of our images into our desired 2D array. We then label each image appropriately.
We then concatenate this all into an X_value and Y_value (features vs labels), and we split this training data.
hogFunction() properly converts our images into data we can use, 2D arrays holding a 1D array per image using a homogenous size for all arrays, so that they are able to be compared and classified.
After splitting the data into training or testing sets, we are ready to build our KNN and K-means clustering models.
Using KNeighborsClassifier, we build a KNN model using our training data.

![Confusion Matrix 1]()

According to our matrix, our model assumed that most images were of leopards.
Our total accuracy using this KNN model is 43%, which isn't practical, but is greater than a 1 in 3 random chance, 33%. In that way, our model is more accurate than a random number generator.
We see that using HOG as our feature, our model has trouble identifying any zebras. They can only differentiate between coyotes and leopards.

## Models 2 : AVERAGE RGB
In this section, we utilize the extractRGBFunction to extract the mean RGB value of an image.
This one is a little different, though. Instead of a 1D array representing our image, we are using one integer value : an average RGB value for all pixels of an image.
The reason why we are trying this out, although lower level, is because our three classifications : zebras, leopards, and coyotes, have distinct colors, and their natural habitats also are very distinct (white snow vs green pastures).

![Confusion Matrix 2]()


## Model 3 : EDGES
In this section, we utilize the cv2 library's ability to extract the edges from an image as its main feature for our models.
As usual, grab the edges feature for each image, and label them correspondingly.
All the images have been converted into 1D arrays depicting the outlines of the images, focusing on patterns and shapes. As we see, zebra stripes are very pronounced, as well as the leopards.
Split the data into training and testing sets.
Build our KNN model with our training data.
As you can see, our accuracy using edges is not very good. This was not what we expected, because the animal outlines seem very distinct from one another with the human eye.
It is no more significant than a random number generator. Why is this? We create a confusion matrix to see our model's accuracies and preferences.
![Confusion Matrix 3]()

As you can see, our model thinks EVERYTHING is a coyote.
We decided to keep this section because we initially banked our hopes onto edges, because zebras and leopards have very specific patterns that we thought would give this model an advantage.
However, the computer who flattens these images anyway compared each pixel as either white or black, and coyotes' images were mostly black, skewing our model.
This helps us realize that we cannot keep it binary, and we decide to try an RGB model, reflecting on our average RGB model and our failed edges model.
You will see this final RGB model near the end of the document.

# K-Means Clustering
## In this section, we use the K-Means clustering algorithm using the hogFunction
![Silhoutte 1]()
Silhouette Score: 0.020696293943144255
K-Means Clustering Analysis: Histogram of Oriented Gradients

In this graph, our average silhouette score was only 0.04 at best. The clusters for our HOG model appeared to be incredibly close to each other. This is likely due to gradients concentrations not having much variation between different classes of animals in our images.
cluster:  0
zebra:  0
coyote:  1
leopard: 0
cluster:  1
zebra:  1
coyote:  1
leopard: 0
cluster:  2
zebra:  22
coyote:  25
leopard: 32
cluster:  3
zebra:  7
coyote:  30
leopard: 13
cluster:  4
zebra:  25
coyote:  7
leopard: 8
cluster:  5
zebra:  0
coyote:  0
leopard: 1
cluster:  6
zebra:  0
coyote:  1
leopard: 0
cluster:  7
zebra:  13
coyote:  7
leopard: 21
cluster:  8
zebra:  9
coyote:  28
leopard: 25
Overall accuracy of the K Means clustering algorithm: 

0.04113178148625973

## In this section, we will use K-Means clustering on the HSV of an image.

![Silhoutte2]()
Silhouette Score: 0.019960438658874913
K-Means Clustering Analysis: Edges
Looking at the K-means clustering for our model using edges, we have a low average silhouette score at 0.11. The data appears to be in one large cluster similar to the HOG model K-means clustering. This is because edges could be anywhere on an image and our model doesn’t try to isolate patterns of edges or what body part they indicate.
cluster:  0
zebra:  41
coyote:  8
leopard: 22


cluster:  1
zebra:  1
coyote:  0
leopard: 0


cluster:  2
zebra:  0
coyote:  0
leopard: 1


cluster:  3
zebra:  32
coyote:  92
leopard: 75


cluster:  4
zebra:  1
coyote:  0
leopard: 0


cluster:  5
zebra:  0
coyote:  0
leopard: 1


cluster:  6
zebra:  0
coyote:  0
leopard: 1


cluster:  7
zebra:  1
coyote:  0
leopard: 0


cluster:  8
zebra:  1
coyote:  0
leopard: 0


Overall accuracy of the K Means clustering algorithm: 
0.10851950171324551
## Our final model : Using RGB Values as our Feature
Now that we have looked at edges and avg RGB as features, we want to see what happens if we base our models off of one feature : RGB values.

We create an extract_RGB_vals() function that grabs an image and extracts their RGB values for each pixel, and flattens that image into a 1D array.
We grab the RGB 1D arrays for each image, and compile them into rgb_features.
With the labels, we split the data into training and testing.
a 224x224 image turns into 224x224x3(RGB) = 150528 dimensions.
Now, we build a KNN model, and see our results.
Accuracy: 0.625
Accuracy for Zebra Images: 0.5294117647058824
As you see, our accuracy for this model is .625, which is significantly higher than an RNG model's .33.

This is most likely due to the contrasting colors in both animals and habitat backgrounds.
We see that this model is accurate in their respective categories. The model does not overdecide on a particular class like our edges model.

Next, we evaluate using a K-Means clustering model.

To find an optimal K, we try the elbow method.
![Elbow Graph]()
K-Means Clustering Analysis: RGB Intensity

Looking at the K-means clustering for the intensity of RGB values, our average silhouette score is again very low, indicating the data points are very close to the decision boundaries between clusters. In this graph, they are overlapping. This is to be expected as the RGB pixels in a single image are heavily varied if the animal we are trying to classify is colorful or the background contrasts the animal. This data seems to have a lot of darker colors, as much of the data is located near the origin. We notice a small cluster near the top right, indicating a high intensity of white, which could indicate a zebra. If the pictures were cropped to just have the animal itself with a blank background, then this would probably provide a more accurate clustering.


# CONCLUSION
For our project, we used multiple features and predictors in order to classify images of animals.
We initially assumed that HOG or edge extraction would help classify animals. However, we find that to not be the case.
The RGB intensity values model turned out to be the most accurate at 0.625 accuracy. This is significantly higher than a random number generator guessing a one in three chance.
The edges model turned out worse than we thought, because to the human eye, the outlines were very distinguishable.
The issue was most likely that the images were binary. Outline is 1, and the rest is 0. This made it so that most of the values were at 0, or black.
Our coyote dataset were filled with black images, so the model began to think that every image was a coyote despite the many white outlines of a zebra pattern.
This may be true for multiple reasons. The biggest reason may be that the habitats of the animals were very different in terms of RGB values. It was very easy for the computer to guess coyotes due to their extremely white snow backgrounds, for example.
KNN models turned out to be better than the K-means clustering. This is due to the fact that unsupervised learning techniques are not optimal for already labeled datasets. This is more for the jobs of supervised learning techniques, such as KNN or neural networks.

Although many of our hypotheses were null, we found success with the RGB intensity model KNN.

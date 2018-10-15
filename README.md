# Object-Counting
My goal is to get the number count of objects in image

1. sift to find descriptors for every image, and stack all the descriptors
2. use the stacked descriptor to train a kmean for clustering
3. create a histogram for every image
4. train a linear model to predict the number count of treelogs

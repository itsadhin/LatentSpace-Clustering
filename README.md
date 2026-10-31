### Latent Space Clustering

The Basic format is gonna be to run LatentSpace_extract.py which contains the VAE to get latent representation. To get the laten space representation for  each new dataset just change the train and test dataset and run the code and save it.

#### Kmeans

Kmeans is a distance based clustering algorithm which sorts points into groups based on points that are closest to the center.
Initiall we have to choose a k value so know what is the best number of cluster that we can get from the data.
Inorder to do it there are a few ways like Elbow method a visual way to find k but not that dependable on higher dimensional
data from my experience. The best to try out are Silhouette Score (Closer to 1 better), Calinski Harabasz Score (Higher the better)
and David Bouldin Score (lower the score better). Try these over a range of kmeans and see what n_components gives the best 
result and choose it.

#### DBSCAN

Density based clustering method, it looks for dense regions of points and separates it from points that are less dense.
We have to choose the radius around which we have to check for clustering and the same radius is used for all clusters
which is a downside cause there maybe clusters of other sizes that get avoided from this. This is where HDSCAN comes to the picture

#### HDBSCAN

Similar to DBSCAN but we dont have to choose the eps it tries to do it all and makes a hierarchy of it and keeps the best clusters.
All that we need to provide is the minimum number of points needed inside the cluster.

Credits: lyeoni for the [VAE](https://github.com/lyeoni/pytorch-mnist-VAE/blob/master/pytorch-mnist-VAE.ipynb)  model that I used here

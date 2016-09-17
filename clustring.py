import tensorflow as tf
from random import*
import random
from numpy import array
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
 
def TFKMeansCluster(vectors, noofclusters):
    """
    K-Means Clustering using TensorFlow.
    'vectors' should be a n*k 2-D NumPy array, where n is the number
    of vectors of dimensionality k.
    'noofclusters' should be an integer.
    """
 
    noofclusters = int(noofclusters)
    assert noofclusters < len(vectors)
 
    #Find out the dimensionality
    dim = len(vectors[0])
 
    #Will help select random centroids from among the available vectors
    vector_indices = list(range(len(vectors)))
    shuffle(vector_indices)
 
    #GRAPH OF COMPUTATION
    #We initialize a new graph and set it as the default during each run
    #of this algorithm. This ensures that as this function is called
    #multiple times, the default graph doesn't keep getting crowded with
    #unused ops and Variables from previous function calls.
 
    graph = tf.Graph()
 
    with graph.as_default():
 
        #SESSION OF COMPUTATION
 
        sess = tf.Session()
 
        ##CONSTRUCTING THE ELEMENTS OF COMPUTATION
 
        ##First lets ensure we have a Variable vector for each centroid,
        ##initialized to one of the vectors from the available data points
        centroids = [tf.Variable((vectors[vector_indices[i]]))
                     for i in range(noofclusters)]
        ##These nodes will assign the centroid Variables the appropriate
        ##values
        centroid_value = tf.placeholder("float64", [dim])
        cent_assigns = []
        for centroid in centroids:
            cent_assigns.append(tf.assign(centroid, centroid_value))
 
        ##Variables for cluster assignments of individual vectors(initialized
        ##to 0 at first)
        assignments = [tf.Variable(0) for i in range(len(vectors))]
        ##These nodes will assign an assignment Variable the appropriate
        ##value
        assignment_value = tf.placeholder("int32")
        cluster_assigns = []
        for assignment in assignments:
            cluster_assigns.append(tf.assign(assignment,
                                             assignment_value))
 
        ##Now lets construct the node that will compute the mean
        #The placeholder for the input
        mean_input = tf.placeholder("float", [None, dim])
        #The Node/op takes the input and computes a mean along the 0th
        #dimension, i.e. the list of input vectors
        mean_op = tf.reduce_mean(mean_input, 0)
 
        ##Node for computing Euclidean distances
        #Placeholders for input
        v1 = tf.placeholder("float", [dim])
        v2 = tf.placeholder("float", [dim])
        euclid_dist = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(
            v1, v2), 2)))
 
        ##This node will figure out which cluster to assign a vector to,
        ##based on Euclidean distances of the vector from the centroids.
        #Placeholder for input
        centroid_distances = tf.placeholder("float", [noofclusters])
        cluster_assignment = tf.argmin(centroid_distances, 0)
 
        ##INITIALIZING STATE VARIABLES
 
        ##This will help initialization of all Variables defined with respect
        ##to the graph. The Variable-initializer should be defined after
        ##all the Variables have been constructed, so that each of them
        ##will be included in the initialization.
        init_op = tf.initialize_all_variables()
 
        #Initialize all variables
        sess.run(init_op)
 
        ##CLUSTERING ITERATIONS
 
        #Now perform the Expectation-Maximization steps of K-Means clustering
        #iterations. To keep things simple, we will only do a set number of
        #iterations, instead of using a Stopping Criterion.
        noofiterations = 50
        for iteration_n in range(noofiterations):
 
            ##EXPECTATION STEP
            ##Based on the centroid locations till last iteration, compute
            ##the _expected_ centroid assignments.
            #Iterate over each vector
            for vector_n in range(len(vectors)):
                vect = vectors[vector_n]
                #Compute Euclidean distance between this vector and each
                #centroid. Remember that this list cannot be named
                #'centroid_distances', since that is the input to the
                #cluster assignment node.
                distances = [sess.run(euclid_dist, feed_dict={v1: vect, v2: sess.run(centroid)}) for centroid in centroids]
                #Now use the cluster assignment node, with the distances
                #as the input
                assignment = sess.run(cluster_assignment, feed_dict = {
                    centroid_distances: distances})
                #Now assign the value to the appropriate state variable
                sess.run(cluster_assigns[vector_n], feed_dict={
                    assignment_value: assignment})
 
            ##MAXIMIZATION STEP
            #Based on the expected state computed from the Expectation Step,
            #compute the locations of the centroids so as to maximize the
            #overall objective of minimizing within-cluster Sum-of-Squares
            for cluster_n in range(noofclusters):
                #Collect all the vectors assigned to this cluster
                assigned_vects = [vectors[i] for i in range(len(vectors))
                                  if sess.run(assignments[i]) == cluster_n]
                #Compute new centroid location
                new_location = sess.run(mean_op, feed_dict={
                    mean_input: array(assigned_vects)})
                #Assign value to appropriate variable
                sess.run(cent_assigns[cluster_n], feed_dict={
                    centroid_value: new_location})

                
 
        #Return centroids and assignments
        centroids = sess.run(centroids)
        assignments = sess.run(assignments)
        return centroids, assignments

clusters=5

points=[]
for i in range(100):
	a=random.uniform(0.0,1024.0)
	b=random.uniform(0.0,1024.0)
	
	point=[a,b]
	points.append(point)
# print points
# a=raw_input()

points_np=array(points)
centroids,assignments=TFKMeansCluster(points_np, clusters)

# print "centroids:",centroids
# print "assignments:",assignments


x=[point[0] for point in points]
y=[point[1] for point in points]

for i in range(clusters):
    temp_list=[points_np[j] for j in range(len(assignments)) if assignments[j]==i]
    temp_list=array(temp_list)
    hull=ConvexHull(temp_list)
    plt.plot(temp_list[hull.vertices,0], temp_list[hull.vertices,1], 'r--', lw=2)


plt.plot(points_np[:,0], points_np[:,1], 'o')
plt.show()
# master = Tkinter.Tk()

# w = Tkinter.Canvas(master, width=1024, height=1024)
# w.pack()
# colors=['forest green', 'olive drab', 'dark khaki', 'khaki', 'pale goldenrod', 'light goldenrod yellow',
#     'light yellow', 'yellow', 'gold', 'light goldenrod', 'goldenrod', 'dark goldenrod', 'rosy brown',
#     'indian red', 'saddle brown', 'sandy brown']
# for i in range(len(points)):
#         w.create_rectangle(points[i][0]-5, points[i][1]-5, points[i][0]+5, points[i][1]+5, fill=colors[assignments[i]])

# Tkinter.mainloop()

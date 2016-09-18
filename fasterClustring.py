
import tensorflow as tf
from random import*
import random
from numpy import array
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull



def kMeansCluster(vector_values, num_clusters, max_num_steps, stop_coeficient = 0.0):
    vectors = tf.constant(vector_values)
    centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors),[0,0],[num_clusters,-1]))

    old_centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors),[0,0],[num_clusters,-1]))
    centroid_distance = tf.Variable(tf.zeros([num_clusters,2]))

    expanded_vectors = tf.expand_dims(vectors, 0)
    expanded_centroids = tf.expand_dims(centroids, 1)

    print expanded_vectors.get_shape()
    print expanded_centroids.get_shape()

    distances = tf.reduce_sum(
      tf.square(tf.sub(expanded_vectors, expanded_centroids)), 2)
    assignments = tf.argmin(distances, 0)

    means = tf.concat(0, [
      tf.reduce_mean(
          tf.gather(vectors,tf.reshape(tf.where(tf.equal(assignments, c)),[1,-1])),reduction_indices=[1])
      for c in range(num_clusters)])

    save_old_centroids = tf.assign(old_centroids, centroids)

    update_centroids = tf.assign(centroids, means)
    init_op = tf.initialize_all_variables()

    #performance = tf.assign(centroid_distance, tf.sub(centroids, old_centroids))
    #check_stop = tf.reduce_sum(tf.abs(performance))

    with tf.Session() as sess:
      sess.run(init_op)
      for step in xrange(max_num_steps):
        print "Running step " + str(step)
        sess.run(save_old_centroids)
        _, centroid_values, assignment_values = sess.run([update_centroids,centroids,assignments])
        #sess.run(check_stop)
        #current_stop_coeficient = check_stop.eval()
        # print "coeficient:", current_stop_coeficient
        # if current_stop_coeficient <= stop_coeficient:
        #   break

      return centroid_values, assignment_values




clusters=4




fig, ax = plt.subplots()
x=[1000.0]
y=[1000.0]
for t in range(100):
	points=[]
	new_x=[]
	new_y=[]
	
        if t == 0:
    	        points_new, = ax.plot(x, y, marker='o', linestyle='None')
    	        ax.set_xlim(0, 1000.0) 
       	        ax.set_ylim(0, 1000.0) 
    	else:
		for b1 in range(15):
			a=random.uniform(0.0,400.0)
			b=random.uniform(400.0,600.0)
			point=[a,b]
			points.append(point)
			new_x.append(a)
			new_y.append(b) 
		for b1 in range(15):
			a=random.uniform(400.0,600.0)
			b=random.uniform(0.0,400.0)
			point=[a,b]
			points.append(point)
			new_x.append(a)
			new_y.append(b) 
		for b1 in range(15):
			a=random.uniform(400.0,600.0)
			b=random.uniform(600.0,1000.0)
			point=[a,b]
			points.append(point)
			new_x.append(a)
			new_y.append(b) 
		for b1 in range(15):
			a=random.uniform(600.0,1000.0)
			b=random.uniform(400.0,600.0)
			point=[a,b]
			points.append(point)
			new_x.append(a)
			new_y.append(b)   		
		#print new_x
    
		points_new.set_data(new_x, new_y)
		points_np=array(points)
		print "printing randomly generated points",points_np

		#centroids,assignments=TFKMeansCluster(points_np, clusters)
		centroids,assignments=kMeansCluster(points_np, 4, 10)


		x_update=[point[0] for point in points]
		y_update=[point[1] for point in points]
		
		for i in range(clusters):
		    temp_list=[points_np[j] for j in range(len(assignments)) if assignments[j]==i]
		    temp_list=array(temp_list)
		    hull=ConvexHull(temp_list)
		    plt.plot(temp_list[hull.vertices,0], temp_list[hull.vertices,1], 'r--', lw=2)
		    plt.scatter(points_np[:,0],points_np[:,1])
		
		plt.grid(True)
		plt.pause(1)
		plt.close()

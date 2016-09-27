import tensorflow as tf
from random import*
import random
from classes import *
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

    # print expanded_vectors.get_shape()
    # print expanded_centroids.get_shape()

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


    with tf.Session() as sess:
      sess.run(init_op)
      for step in xrange(max_num_steps):
        # print "Running step " + str(step)
        sess.run(save_old_centroids)
        _, centroid_values, assignment_values = sess.run([update_centroids,centroids,assignments])
        

      return centroid_values, assignment_values


def calc_cluster(rangeX,rangeY,max_iter,num_clust):
  points=[]
  cars=[]
  cluster_list=[]
  total_lane_weight=[]
  for num_rand_cars in range(int(random.uniform(4.0,100.0))):     #creates random cars
			
    a=random.uniform(rangeX[0],rangeX[1])
    b=random.uniform(rangeY[0],rangeY[1])
    vel=random.uniform(0.0,10.0)
    point=[a,b]
    points.append(point)
    cars.append(car(point,vel))
  points_np=array(points)

  centroids,assignments=kMeansCluster(points_np, num_clust, max_iter) 

  for i in range(num_clusters):
    for j in range(len(assignments)):
      if assignments[j]==i:
        cars[j].cluster=i

  
  for i in range(num_clusters):
    temp_list=[points_np[j] for j in range(len(assignments)) if assignments[j]==i]

    cars_in_cluster=[cars[k] for k in range(len(assignments)) if cars[k].cluster==i ] 
    #extracting cars which belongs to ith cluuster


    temp_cluster=cluster(cars_in_cluster,centroids[i],i)
    temp_cluster.assign_avg_vel() 
    total_lane_weight.append(temp_cluster.assign_weight(200.0))     #append cluster weight in lane weight vector
    #print total_lane_weight

    cluster_list.append (temp_cluster)
    # print temp_cluster

    if len(temp_list)>3:
      temp_list=array(temp_list)
      hull=ConvexHull(temp_list)
      plt.plot(temp_list[hull.vertices,0], temp_list[hull.vertices,1], 'r--', lw=2)
    plt.scatter(points_np[:,0],points_np[:,1])

  return cluster_list,sum(total_lane_weight)

num_clusters=4
max_iter=50
fig, ax = plt.subplots()
x=[1000.0]
y=[1000.0]
for t in range(100):
		
  if t == 0:
    points_new, = ax.plot(x, y, marker='o', linestyle='None')
    ax.set_xlim(0, 1000.0)
    ax.set_ylim(0, 1000.0)
  else:
    lane_cluster_list=[]
    # plt.close()
    lane_cluster_list.append(calc_cluster([0.0,400.0],[500.0,600.0],max_iter,num_clusters))

    # print lane_cluster_list[0][0]
    #raw_input()
    lane_cluster_list.append(calc_cluster([600.0,1000.0],[400.0,500.0],max_iter,num_clusters))
    # plt.grid(True)
    # plt.pause(1)
    # #raw_input()
    lane_cluster_list.append(calc_cluster([400.0,500.0],[0.0,400.0],max_iter,num_clusters))
    # plt.grid(True)
    # plt.pause(1)
    # #raw_input()
    lane_cluster_list.append(calc_cluster([500.0,600.0],[600.0,1000.0],max_iter,num_clusters))
    # plt.grid(True)
    # plt.pause(1)
    

    time_durations=[]
    time_durations.append(distribute_time(180,[x[1] for x in lane_cluster_list]))
   
    print time_durations,"printing final"
    raw_input()

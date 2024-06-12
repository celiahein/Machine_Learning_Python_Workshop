# The first example that uses a linear model to classify two blobs (or clusters)
# of dots in 2D feature space. The main goal of the example is to help the 
# audience to understand
#     * the basic concepts like training samples, features, feature space,
#       trainable parameters and parameter space, etc.
#     * machine learning is an optimization problem
#     * how gradient descent is used to find the "global" minimum (with animated illustration)
#
# Written by Weiguang Guan
# Sharcnet/Digital Research Alliance of Canada
# Dec, 2022

import sklearn.datasets as skd
import sklearn.model_selection as skms

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import random
import time
import math

###############################################################################
# NOTES: 
# Here we use a linear model:
#         output = ax+by+c

###############################################################################
# Calculate the intersections between ax+by+c=0 and bounding box "bbox"
# bbox = [xmin, xmax, ymin, ymax]
def end_points(a, b, c, bbox) :
    if (abs(a)>abs(b)) :
        X = [-(b*(bbox[2])+c)/a, -(b*bbox[3]+c)/a]
        Y = [bbox[2], bbox[3]]
    else :
        X = [bbox[0], bbox[1]]
        Y = [-(a*(bbox[0])+c)/b, -(a*bbox[1]+c)/b]
    return X, Y

# Draw the line defined by a, b, c within bounding box "bbox"
def draw_line(a, b, c, bbox) :
    X, Y = end_points(a, b, c, bbox)
    line, = ax.plot(X, Y, marker = 'o', color="green")

    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])

    return line

###############################################################################
# Main
np.random.seed(2022)
tf.compat.v1.disable_eager_execution()

# Generate two clusters of dots
SCALE = 2.0
dots, labels = skd.make_blobs(500, centers=2, center_box=(-SCALE, SCALE))
# shape of dots = (500, 2), shape of lables = (500)

# Split into training and testing datasets
train_xy, test_xy, train_l, test_l = skms.train_test_split(dots, labels, test_size = 0.3)
# shape of train_xy = (350, 2)

# Plot the training data
plt.ion()

figure, ax = plt.subplots(figsize=(8, 8))

for i in range(train_xy.shape[0]):
    if (train_l[i]==1) :
        ax.plot(train_xy[i,0], train_xy[i,1], 'o', color="red")
    else :
        ax.plot(train_xy[i,0], train_xy[i,1], 'o', color="blue")

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Feature space (x, y)', fontweight ="bold")

input("Hit the return key to continue!")

left, right = ax.get_xlim()
bottom, top = ax.get_ylim()

# Randomly initialize a, b, c 
abc_rnd = np.random.normal(0.0, 1.0, 3)
abc = tf.Variable(abc_rnd.astype('float32'))

# Draw line ax+by+c=0 (decision boundary)
line = draw_line(abc_rnd[0], abc_rnd[1], abc_rnd[2], [left, right, bottom, top])

figure.canvas.draw()
figure.canvas.flush_events()

time.sleep(2.0)
input("Hit the return key to continue!")

# Define the model
xy = tf.compat.v1.placeholder(dtype=tf.float32)
l = tf.compat.v1.placeholder(dtype=tf.float32)

#output = tf.tensordot(xy, abc[:2], 1) + abc[2]
output = abc[0]*xy[:,0] + abc[1]*xy[:,1] + abc[2]   # output=ax+by+c
output = tf.math.sigmoid(output)

# Define the loss function and accuracy
loss = tf.reduce_mean(tf.square(output - l)) # MSE
#loss = tf.reduce_mean(-(l*tf.math.log(output) + (1-l)*tf.math.log(1.0-output))) # entropy

accuracy = tf.reduce_mean(tf.math.abs(tf.math.sign(output-0.5)+2.0*l-1.0)) / 2.0

# Define the gradient descent optimizer
optimizer = tf.compat.v1.train.GradientDescentOptimizer(1.0)
optm = optimizer.minimize(loss)

# Initialization before training iterations
init = tf.compat.v1.global_variables_initializer()
sess = tf.compat.v1.Session()
sess.run(init) # reset values to a state for the training to start with

ITERATIONS = 100

# Store the values of a,b,c through training iterations so that we can
# display the footprints of the training process in the parameter space.
abc_array = np.zeros((ITERATIONS+1, 3)) 

cur_abc = sess.run(abc)
abc_array[0,:] = cur_abc

# Iterative training process
for i in range(ITERATIONS):
    sess.run(optm, {xy:train_xy, l:train_l})

    cur_abc = sess.run(abc)
    abc_array[i+1,:] = cur_abc 

    X, Y = end_points(cur_abc[0], cur_abc[1], cur_abc[2], [left, right, bottom, top])
    line.set_xdata(X)
    line.set_ydata(Y)

    figure.canvas.draw()
    figure.canvas.flush_events()
    time.sleep(0.05)

    # Evaluate error and accuracy the model
    cur_loss, cur_accuracy = sess.run([loss, accuracy], {xy:train_xy, l:train_l})
    print("loss = %s, accuracy = %s"%(cur_loss, cur_accuracy))

# Evaluate error and accuracy of the model using testing data
cur_loss, cur_accuracy = sess.run([loss, accuracy], {xy:test_xy, l:test_l})
print("Using test data: loss = %s, accuracy = %s"%(cur_loss, cur_accuracy))

# Show the training process in the parameter space
RHO_RANGE = 200
RHO_SCALE = 40.0
THETA_RANGE = 360
loss_map = np.zeros((2*RHO_RANGE, THETA_RANGE))

for j in range(THETA_RANGE) :
    Wnumpy = np.array([math.cos(j*np.pi/180.0), math.sin(j*np.pi/180.0)]);
    out1 = train_xy.dot(Wnumpy)
    for i in range(-RHO_RANGE, RHO_RANGE) :
        out2 = out1 - i/RHO_SCALE
        out2 = 1.0/(1.0 + np.exp(-out2))
        sum2 = np.square(out2-train_l).sum()
        loss_map[i+RHO_RANGE,j] = sum2

input("Hit the return key to continue!")

# Plot the loss map
z_min, z_max = loss_map.min(), loss_map.max()

ax.cla()
extent = 0, 360, -RHO_RANGE, RHO_RANGE
c = ax.imshow(loss_map, cmap ='plasma', vmin=z_min, vmax=z_max, 
                    interpolation ='nearest', extent=extent, origin ='lower')
plt.colorbar(c)

plt.xlabel('Theta (in degrees)')
plt.ylabel('Rho (scaled 40 times)')

figure.canvas.draw()
figure.canvas.flush_events()

plt.title('Parameter space (theta, rho)', fontweight ="bold")

# Training track in the parameter space
input("Hit the return key to show training track in the parameter space!")

norm = np.sqrt(np.square(abc_array[:,0]) + np.square(abc_array[:,1]))

rho = -abc_array[:,2]/norm

for i in range(ITERATIONS+1) :
    theta = math.atan2(abc_array[i,1], abc_array[i,0])
    if (theta<0) :
        theta += 2.0*np.pi

    theta *= 180.0/np.pi

    ax.plot(theta, rho[i]*RHO_SCALE, 'o', color="red")
    figure.canvas.draw()
    figure.canvas.flush_events()

input("Hit the return key to finish!")

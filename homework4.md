# Homework 4: Implementing + Evaluating Spatial Partitioning Data Structures

## Due June 19

In this lab you'll implement and compare the performance of the 3 spatial partitioning data structures we've discussed in class: bucketing, KD-trees, and QuadTrees

## Part 1: Implementation

Implement the 3 data structures.

Each of these should have an appropriate constructor, rangeQuery(p, r), and KNN(p, k) methods.

I've provided a lot of skeleton code [here](spatialDataStructures)

C++ notes:

* The Point's dimension is a template parameter.  This mostly doesn't matter, except that you can only fill it in with something you know/can compute at compile time.  (You can't fill it in with a variable, for example).  You can use Dimension to get its value inside any of the structs/functions that have it as a template parameter.

* The C++ stdlib has a bunch of useful functions that will make your life a lot easier: std::partition, std::nth_element, std::push_heap, std::pop_heap, std::clamp to name a few.  I'd recommend using std::unique_ptr to store any tree nodes.  You don't have to worry aboud delete-ing unique_ptr's!

* Debug your data structures by operating on a small set of points so you can understand what's happening!

* Start early!  There are a bunch of tricky aspects to the code

## Part 2: Testing

Test the KNN method of your implementations over as wide of a range of conditions as you can (reasonably) do.  You should be able to run your data-generating code in ~5 minutes.  We're interested in the running time per KNN query and how it is affected by:

* k
* N (the total number of points)
* D (the dimension of your data.  Quadtrees will only work for D=2)

You may also want to play with different approaches for choosing the number of bucket divisions (for bucketing), or the maximum leaf size (for your quadtree).  This is optional, but pick something reasonable for these parameters (play around a little bit and pick the best optin you find).

Your testing code should create a CSV file that contains all the necessary data to load into a DataFrame in a Jupyter notebook.

Perform your tests for uniformly distributed data points, and gaussian distributed data points (which will clump them around the mean)

To reduce noise, I recommend timing several KNN queries (10, or 100) and using the average time.  Repeating each experiment 3 times or so (with different random points) should also help you feel more confident about your results.



## Part 3: Analysis

Analyze the data you collected.  

* Plot parts of your data to make sense of it(what impact to K, N, D, and the data structure have?)
* Perform regression based on the performance we expect to see.  Do tests confirm or disprove our expectations?
* Are there any aspects of your data that seem unusual?  Can you explain them?





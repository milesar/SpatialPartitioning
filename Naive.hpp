/** Naive, inefficient ( O(N) ) but simple routines for performing exhaustive
 * KNN and range searches over a collection of points of d-dimensions.
 */
#pragma once

#include <array>
#include <ostream>
#include <cmath>
#include <vector>
#include "Point.hpp"

/** Performs a terribly inefficient (but easy to verify) k nearest-neighbor search on an array
 * of points for a target point, to provide a validation case for other, more complicated methods.
 *
 * @tparam Dimension the number of dimensions (d) in the points space.
 * @param points an array of points of dimension d.
 * @param target the target point around which to perform the knn search.
 * @param k the number of nearest neighbors to locate.
 * @return an array of points, the nearest neighbors, sorted by euclidean distance.
 */
template<int Dimension>
std::vector<Point<Dimension> > naive_knn(const std::vector<Point<Dimension> > &points,
                                   const Point<Dimension> &target, int k) {

    std::vector<Point<Dimension> > pointsCopy = points;
    DistanceComparator sort_lambda(target);

    std::sort(pointsCopy.begin(), pointsCopy.end(), sort_lambda);

    std::vector<Point<Dimension> > knn(pointsCopy.begin(), pointsCopy.begin() + k);
    return knn;
}

/** Helper method for comparing two floats with a cutoff, epsilon.
 *
 * @param lhs 1st float
 * @param rhs 2nd float
 * @param epsilon precision cutoff
 * @return true if the difference between the two floats is less than epsilon.
 */
bool cmpf(float lhs, float rhs, float epsilon = 0.005f)
{
    return (fabs(lhs - rhs) < epsilon);
}

/** Performs a brute force search for points within a range (r, radius).
 *
 * @tparam Dimension the number of dimensions (d) in the points space.
 * @param points an array of points of dimension d.
 * @param p the target point around which to search for neighbors.
 * @param radius the distance within which to find finds.
 * @return an array of all points within the specified radius.
 */
template<int Dimension>
std::vector<Point<Dimension> > naive_range(const std::vector<Point<Dimension> > &points,
                                          const Point<Dimension> &target, float radius) {

    std::vector<Point<Dimension> > in_range;

    for(const auto & point : points) {
        float dist = distance(point, target);
        if( dist <= radius && !cmpf(dist, radius)) {
            in_range.push_back(point);
        }
    }

    return in_range;
}


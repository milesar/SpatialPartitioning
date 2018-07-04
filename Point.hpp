/** Methods and objects for handling individual instances and collections
 * of d-dimensional points.
 *
 */

#pragma once

#include <array>
#include <ostream>
#include <cmath>
#include <algorithm>

/** Provides a Point object, of dimension d.
 *
 * Read only, and no constructor is provided in this implementation.
 *
 * @tparam Dimension the dimension space of the points.
 */
template<int Dimension>
struct Point {
    //Fixed size, inline array
    std::array<float, Dimension> point;

    //index points directly.  This is read only.
    float operator[](int index) const { return point[index]; }

};

/** Calculates the euclidean distance between two points in d-dimensions.
 *
 * Could be further optimized, by returning and working with the squared distance
 * (the explicit distance is unnecessary for comparison as long as we are consistent,
 * and requires one less costly operation (math.sqrt) )
 *
 * @tparam Dimension space of the points.
 * @param a the first point.
 * @param b the second point.
 * @return the distance between the two points.
 */
template<int Dimension>
float distance(const Point<Dimension> &a, const Point<Dimension> &b) {
    float dist = 0;
    for (int i = 0; i < Dimension; ++i) {
        dist += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return std::sqrt(dist);
}

/** Method for printing a point to the terminal. Debug utility.
 *
 * @tparam Dimension project dimension space.
 * @param outs output stream.
 * @param p point to print.
 * @return the output stream.
 */
template<int Dimension>
std::ostream &operator<<(std::ostream &outs, const Point<Dimension> &p) {
    outs << "Point(";
    for (int i = 0; i < Dimension - 1; ++i) {
        outs << p[i] << ", ";
    }
    outs << p[Dimension - 1] << ")";
    return outs;
}

/** Comparator that provides lexicographical ordering of points, by the specified dimension.
 *
 * Usage:
 *
 *      CompareBy<1> compareByY;
 *      std::sort(points.begin(), points.end(), compareByY);
 *
 * The template parameter must be known at compile time!
 *
 * @tparam Dimension
 */
template<int Dimension>
struct CompareBy {

    template<int PD>
    bool operator()(const Point<PD> &lhs, const Point<PD> &rhs) {
        static_assert(Dimension < PD, "must sort by a dimension that exists!");
        // This routine provides lexographic ordering, proceeding after the initialization dimension.
        // Proceeds through each dimension if equal until a difference is encountered.

        if (lhs[Dimension] == rhs[Dimension]) {
            for (int i = 0; i < PD; ++i) {
                if (i != Dimension) {
                    if (lhs[i] != rhs[i]) {
                        return lhs[i] < rhs[i];
                    }
                }
            }
            return false; // points are equal

        } else { // easy case
            return lhs[Dimension] < rhs[Dimension];
        }
    }

};

/** Provides a comparator for the distance between one point and another.
 *
 * @tparam Dimension the current dimension space.
 */
template<int Dimension>
struct DistanceComparator {
public:
    explicit DistanceComparator(const Point<Dimension> &q_)
            : p{q_} {}

    bool operator()(const Point<Dimension> &lhs, const Point<Dimension> &rhs) {
        return distance(p, lhs) < distance(p, rhs);
    }

private:
    Point<Dimension> p;
};

/** Provides an axis-aligned bounding box and a method for finding and returning
 * the closest point in the box.
 *
 * @tparam Dimension the current dimension space.
 */
template<int Dimension>
struct AABB {
    std::array<float, Dimension> mins, maxs;

    // default constructor is infinite, which is useful for KDTree
    AABB() {
        for (int i = 0; i < Dimension; ++i) {
            mins[i] = std::numeric_limits<float>::min();
            maxs[i] = std::numeric_limits<float>::max();
        }
    }

    // returns the point in the bounding box (likely on the boundary) that is closest to p
    Point<Dimension> closestInBox(const Point<Dimension> &p) {

        std::array<float, Dimension> arr;
        for (int i = 0; i < Dimension; ++i) {
            arr[i] = std::clamp(p[i], mins[i], maxs[i]);
        }
        return Point<Dimension>{arr};

    }
};

/** Finds and returns the closest bounding box for a collection of points.
 *
 * @tparam Dimension the current dimension space.
 * @param points a collection of point objects.
 * @return a bounding box containing the specified collection of points.
 */
template<int Dimension>
AABB<Dimension> getBounds(const std::vector<Point<Dimension> > &points) {

    std::array<float, Dimension> mins, maxs;
    for (int i = 0; i < Dimension; ++i) {
        mins[i] = std::numeric_limits<float>::max();
        maxs[i] = std::numeric_limits<float>::min();
    }
    for (const auto &p : points) {
        for (int i = 0; i < Dimension; ++i) {
            mins[i] = std::min(mins[i], p[i]);
            maxs[i] = std::max(maxs[i], p[i]);
        }
    }

    AABB<Dimension> ret;
    ret.mins = mins;
    ret.maxs = maxs;
    return ret;
}

/** Utility for printing out the contents of a bounding box.
 *
 * @tparam Dimension
 * @param outs an output stream.
 * @param aabb a bounding box object.
 * @return the output stream
 */
template<int Dimension>
std::ostream &operator<<(std::ostream &outs, const AABB<Dimension> &aabb) {
    outs << "AABB( " << Point<Dimension>{aabb.mins} << " : " << Point<Dimension>{aabb.maxs} << " ) ";
    return outs;
}

/** utility for printing out the contents of a point list.
 *
 * @tparam Dimension the dimension of the collection of points.
 * @param points the points to print.
 */
template<int Dimension>
void print_points(const std::vector<Point<Dimension> > &points) {
    for (auto &point : points) {
        std::cout << point << "\n";
    }
}

template<int Dimension>
void print_results(const std::vector<Point<Dimension> > &naive,
                   const std::vector<Point<Dimension> > &test) {
    for (int i = 0; i < naive.size(); i++) {
        std::cout << "\nNaive:\t" << naive[i] << "\nTest:\t" << test[i] << "\n";
    }
}

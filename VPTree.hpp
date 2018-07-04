/** Basic implementation (no insert and delete) of a KDTree
 *
 */
#pragma once

#include "Point.hpp"
#include <memory>
#include <queue>
#include <utility>
#include <algorithm>
#include <iomanip>

template<int Dimension>
class VPTree {
public:
    explicit VPTree(const std::vector<Point<Dimension> > &points) {
        std::vector<Point<Dimension> > pointsCopy = points;
        root = std::unique_ptr<Node>(new Node(pointsCopy.begin(), pointsCopy.end(), size));
    }

    // Range query method, returns all points that are within the specified radius
    // of the target point.
    std::vector<Point<Dimension> > range_query(const Point<Dimension> &p, float radius) const {
        std::vector<Point<Dimension> > in_range;
        range_recursive(root, p, radius, in_range);
        return in_range;
    }

    // KNN query method, returns the set of k points that are closest to the target point.
    std::vector<Point<Dimension> > knn(const Point<Dimension> &p, int k) const {
        std::vector<Point<Dimension> > knn;
        float tau = MAXFLOAT;
        knn_recursive(root, p, k, knn, tau);

        return knn;
    }

    // helper method, returns the size of the tree for validation testing.
    int getSize() {
        return size;
    }

    // helper method, returns the number of nodes visited for validation and efficiency testing.
    int getNodesVisited() {
        return nodes_visited;
    }

    // helper method, resets the number of nodes visited after a search completes.
    int reset_visited() {
        nodes_visited = 0;
    }

    // helper method, prints out the contents of the current tree.
    // NOT ADVISED FOR TREE SIZE > 100.
    void print() {
        postorder(root, 0);
    }

private:
    int size = 0;
    int nodes_visited = 0;

    /** the left hand side of the tree will have all points inside the radius mu for the chosen
     * vantage point, and the right hand side contains all points within the radius mu.
     *
     * @tparam SplitDimension the splitting dimension
     */
    struct Node {
        Point<Dimension> vp;
        float mu;
        std::unique_ptr<Node> inside, outside;

        template<typename Iter>
        Node(Iter begin, Iter end, int &size) {
            size++;
            Iter vantage = begin + (end - begin) / 2;
            vp = *vantage;
            mu = 0.0;
            if (end - begin == 1) {
                return;
            }

            std::swap(*begin, *vantage);

            DistanceComparator<Dimension> lambda(*begin);
            Iter middle = begin + (end - begin) / 2;
            std::nth_element(begin + 1, middle, end, lambda);

            mu = distance(*begin, *middle);
            if ((begin + 1) != middle) {
                inside = std::unique_ptr<Node>(new Node(begin + 1, middle, size));
            }
            if (middle != end) {
                outside = std::unique_ptr<Node>(new Node(middle, end, size));
            }
        }
    };

// The root node.
    std::unique_ptr<Node> root;

/** Recursive range search, helps perform a modified depth first search of the vptree
 * to find all points within the specified range of the tree.
 * @tparam SplitDimension the dimension of points in this search context.
 * @param sub the root node of the tree to search.
 * @param p the target point around which to perform the search.
 * @param radius the radius within which to find all neighbors.
 * @param in_range collection of the points in range encountered thus far..
 */
    void range_recursive(const std::unique_ptr<Node> &sub,
                         const Point<Dimension> &p,
                         float radius,
                         std::vector<Point<Dimension> > &in_range) const {

        // Check if the current point is within the search radius, and add it to the
        // set of points in range.
        auto dist = distance(sub->vp, p);
        if (dist <= radius) {
            in_range.push_back(sub->vp);
        }

        // If the current node has a left node, and its coordinate in this dimension
        // overlaps the search radius, then go left.
        if (sub->inside) {

            if (dist <= sub->mu + radius) {
                range_recursive(sub->inside, p, radius, in_range);
            }
        }

        // If the current node has an outside node, then go
        if (sub->outside) {

            if (dist >= sub->mu - radius) {
                range_recursive(sub->outside, p, radius, in_range);
            }
        }
    };

/** Recursive, modified depth first search method to find the collection of k points
 *
 * @tparam SplitDimension the dimension of points in this search context.
 * @param sub the root node of the tree to search.
 * @param p the target point around which to perform the search.
 * @param k the number of nearest neighbors to find.
 * @param nn collection of the nearest neighbors encountered thus far in the search.
 */
    void knn_recursive(const std::unique_ptr<Node> &sub,
                       const Point<Dimension> &p,
                       int k,
                       std::vector<Point<Dimension> > &nn, float &tau) const {

        auto dist = distance(sub->vp, p);
        // If the list of nearest neighbors, add this point to the list.
        if (nn.size() < k) {
            nn.push_back(sub->vp);
            std::push_heap(nn.begin(), nn.end(), DistanceComparator<Dimension>(p));
        }
            // Otherwise, if the list is full and the current point is closer to the target
            // than the worst point (furthest) in our list, then remove that point and replace
            // it with the current, closer point.
        else if (distance(sub->vp, p) < tau) {

            std::pop_heap(nn.begin(), nn.end(), DistanceComparator<Dimension>(p));
            nn.pop_back();
            nn.push_back(sub->vp);
            std::push_heap(nn.begin(), nn.end(), DistanceComparator<Dimension>(p));
        }

        // Recurse left if the left sub-tree bounding box might contain points
        // that are closer than the worst point currently in our list of neighbors.

        tau = distance(nn.front(), p);
        if (sub->inside) {
            if (dist <= tau + sub->mu || (nn.size() < k)) {
                knn_recursive(sub->inside, p, k, nn, tau);
            }
        }

        // Recurse right if the right sub-tree bounds might contain points
        // that are closer than the worst point currently in our list of neighbors.
        if (sub->outside) {
            if (dist >= tau - sub->mu || (nn.size() < k)) {
                knn_recursive(sub->outside, p, k, nn, tau);
            }
        }
    };

    /** helper function for visualizing the tree contents at the command line.
     *
     * It is advised to only use this method with trees of size <100, as the results
     * are pretty unintelligible thereafter unless you are just looking at the root.
     *
     * @param sub root of the tree to display.
     * @param indent distance to indent with each subtree block.
     */
    void postorder(const std::unique_ptr<Node> &sub, int indent)
    {
        if(sub) {
            if(sub->outside) {
                postorder(sub->outside, indent+4);
            }
            if (indent) {
                std::cout << std::setw(indent) << ' ';
            }
            if (sub->outside) std::cout<<" /\n" << std::setw(indent) << ' ';
            std::cout<< sub->vp<< sub->mu << "\n ";
            if(sub->inside) {
                std::cout << std::setw(indent) << ' ' <<" \\\n";
                postorder(sub->inside, indent+4);
            }
        }
    }
};


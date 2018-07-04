/** Basic implementation (no insert and delete) of a KDTree
 *
 */
#pragma once

#include "Point.hpp"
#include <memory>
#include <queue>
#include <utility>
#include <algorithm>

template<int Dimension>
class KDTree {
public:
    explicit KDTree(const std::vector<Point<Dimension> > &points) {
        std::vector<Point<Dimension> > pointsCopy = points;
        root = std::unique_ptr<Node<0> >(new Node<0>(pointsCopy.begin(), pointsCopy.end(), size));
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
        AABB<Dimension> box;
        knn_recursive(root, p, k, knn, box);

        return knn;
    }

    // Helper method, returns the size of the tree for validation testing.
    int getSize() {
        return size;
    }

    int getNodesVisited() {
        return nodes_visited;
    }

    int reset_visited(){
        nodes_visited = 0;
    }

private:
    int size = 1;
    int nodes_visited = 0;

    template<int SplitDimension> //Don't store the split dimension explicitly

    struct Node {
        Point<Dimension> p;
        //The children will have the "next" splitting dimension
        //Since this is part of the type, we can't "forget" to set it properly... nice!
        std::unique_ptr<Node<(SplitDimension + 1) % Dimension> > left, right;

        // Iterators used to describe the chunk of the points array that belong to
        // this node/subtree.

        template<typename Iter>
        Node(Iter begin, Iter end, int &size) {
            //Our children (if we have any) will use the "next" splitting dimension
            using ChildType = Node<(SplitDimension + 1) % Dimension>;
            Iter middle = begin + (end - begin) / 2;
            std::nth_element(begin, middle, end, CompareBy<SplitDimension>());
            p = *middle;
            if (begin != middle) {
                left = std::unique_ptr<ChildType>(new ChildType(begin, middle, size));
                size++;
            }
            if (middle + 1 != end) {
                right = std::unique_ptr<ChildType>(new ChildType(middle + 1, end, size));
                size++;
            }
        }
    };

    // The root node.
    std::unique_ptr<Node<0> > root;

    // Recursive range search, helps perform a modified depth first search of the kdtree
    // to find all points within the specified range of the tree.
    template<int SplitDimension>
    void range_recursive(const std::unique_ptr<Node<SplitDimension>> &sub,
                         const Point<Dimension> &p,
                         float radius,
                         std::vector<Point<Dimension> > &in_range) const {

        // Check if the current point is within the search radius, and add it to the
        // set of points in range.

        if (distance(sub->p, p) <= radius) {

            in_range.push_back(sub->p);
        }

        // If the current node has a left node, and its coordinate in this dimension
        // overlaps the search radius, then go left.
        if (sub->left) {

            if (sub->p[SplitDimension] >= p[SplitDimension] - radius) {
                range_recursive(sub->left, p, radius, in_range);
            }
        }

        // If the current node has a right node, and its coordinate in this dimension
        // overlaps the search radius, then go right.
        if (sub->right) {

            if (sub->p[SplitDimension] <= p[SplitDimension] + radius) {
                range_recursive(sub->right, p, radius, in_range);
            }
        }
    };

    // Recursive, modified depth first search method to find the collection of k points
    // closest to the target point.
    template<int SplitDimension>
    void knn_recursive(const std::unique_ptr<Node<SplitDimension> > &sub,
                       const Point<Dimension> &p,
                       int k,
                       std::vector<Point<Dimension> > &nn,
                       AABB<Dimension> &box) const {

        // If the list of nearest neighbors needs points, add this point to the list.
        if (nn.size() < k) {
            nn.push_back(sub->p);
            std::push_heap(nn.begin(), nn.end(), DistanceComparator<Dimension>(p));
        }

        // Otherwise, if the list is full and the current point is closer to the target
        // than the worst point (furthest) in our list, then remove that point and replace
        // it with the current, closer point.
        else if (distance(sub->p, p) < distance(p, nn.front())) {

            std::pop_heap(nn.begin(), nn.end(), DistanceComparator<Dimension>(p));
            nn.pop_back();
            nn.push_back(sub->p);
            std::push_heap(nn.begin(), nn.end(), DistanceComparator<Dimension>(p));
        }

        // Recurse left if the left sub-tree bounding box might contain points
        // that are closer than the worst point currently in our list of neighbors,
        // or we need more points.
        if (sub->left) {
            AABB leftbox = box;
            leftbox.maxs[SplitDimension] = sub->p[SplitDimension];
            if ((distance(leftbox.closestInBox(p), p) < distance(nn.front(), p))
                || (nn.size() < k)) {
                knn_recursive(sub->left, p, k, nn, leftbox);
            }
        }

        // Recurse right if the right sub-tree bounds might contain points
        // that are closer than the worst point currently in our list of neighbors,
        // or we need more points.
        if (sub->right) {
            AABB rightbox = box;
            rightbox.mins[SplitDimension] = sub->p[SplitDimension];
            if ((distance(rightbox.closestInBox(p), p) < distance(nn.front(), p))
                || (nn.size() < k)) {
                knn_recursive(sub->right, p, k, nn, rightbox);
            }
        }
    };
};


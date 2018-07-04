/** Basic implementation (no insert and delete) of a PR Quad Tree (Point-Region)
 *
 */
#pragma once

#include "Point.hpp"
#include <memory>
#include <queue>
#include <utility>
#include <vector>

/** the quadtree subdivides the set of points into 4 bounded regions. If points are found within the
 * bounding box defined by each child's quadrant, another node is created in which 4 more quadrants
 * are defined.
 *
 * the orientation and names of each quadrant will follow the naming scheme diagrammed here:
 *
 *          * * * * * * * * * * *
 *          *         *         *
 *          *    1    *    0    *
 *          *         *         *
 *          * * * * * * * * * * *
 *          *         *         *
 *          *    2    *    3    *
 *          *         *         *
 *          * * * * * * * * * * *
 *
 * the 4 tree node array positions correspond with the quadrants for transparency.
 */

class QuadTree {
public:
    explicit QuadTree(const std::vector<Point<2> > &points, int &max_leaves) {
        std::vector<Point<2> > pointsCopy = points;
        AABB<2> box = getBounds(pointsCopy);
        root = std::unique_ptr<Node>(new Node(pointsCopy.begin(),
                                              pointsCopy.end(),
                                              box, max_leaves,
                                              size, height));
    }

    // Range query method, returns all points that are within the specified radius
    // of the target point.
    std::vector<Point<2> > range_query(const Point<2> &p, float radius) const {
        std::vector<Point<2> > in_range;
        range_recursive(root, p, radius, in_range);
        return in_range;
    }

    // KNN query method, returns the set of k points that are closest to the target point.
    std::vector<Point<2> > knn(const Point<2> &p, int k) const {
        std::vector<Point<2> > knn;
        std::unique_ptr<Node> *closest;

        knn_recursive(root, p, k, knn);

        return knn;
    }

    // Helper method, returns the size of the tree for validation testing.
    int getSize() {
        return size;
    }

    int getHeight() {
        return height;
    }


private:
    /** each node in the tree will have a maximum of 4 children, representing the 4 quadrants
     * in the current bounding box of the tree.
     *
     * a node will be null and flagged as a leaf node if there are no points in the range described
     * by that box. if there are less than max_leaves, the node will also be flagged as a leaf node
     * and will contain a list of points found within the bounds of the quadrant. otherwise the node
     * will point to a new quadtree node.
     *
     * nodes are referred to as specified above (one - three).
     *
     * each quadrant can be partitioned over the current range of points into 4 quadrants,
     * first by splitting the vector by x coordinate and then splitting both of the resulting
     * halves by y coordinate.
     *
     * rough diagram of the resulting iterators and quadrants described by each node.
     *
     *         begin      y_lmid      x_mid       y_rmid       end
     *           |          |           |           |           |
     *                 2           1           3           0
     *
     */
    struct Node {
        std::vector<Point<2> > node_points;     // node_points, populated with points in bounds if a leaf node.
        bool leaf = true;                       // the node is a leaf unless we have more than max_leaves in bounds.
        std::unique_ptr<Node> quads[4];         // a vector of pointers to child nodes (if not a leaf node).
        AABB<2> bounds;

        template<typename Iter>
        Node(Iter begin, Iter end, AABB<2> box, const int &max_leaves, int &size, int &height) {

            bounds = box;

            // if we have less points than the max_leaves, we have a leaf node and can add the points to the node.
            if (end - begin <= max_leaves) {
                node_points.assign(begin, end);
                size += node_points.size();
            }
                // if we have more points than the max_leaves, we need to partition this quadrant into 4 sub-quadrants.
            else {
                leaf = false;
                auto x_split = ((box.maxs[0] + box.mins[0]) / 2);
                Iter x_mid = std::partition(begin, end, [x_split](const Point<2> &p) {
                    return p[0] < x_split;
                });

                auto y_split = ((box.maxs[1] + box.mins[1]) / 2);
                Iter y_lmid = std::partition(begin, x_mid, [y_split](const Point<2> &p) {
                    return p[1] < y_split;
                });
                Iter y_rmid = std::partition(x_mid, end, [y_split](const Point<2> &p) {
                    return p[1] < y_split;
                });

                // recursively define and build each quadrants representative node.
                // the zero quadrant node definition.
                if (end != y_rmid) {
                    AABB<2> zero = box;
                    zero.mins[0] = x_split;
                    zero.mins[1] = y_split;

                    quads[0] = std::unique_ptr<Node>(new Node(y_rmid, end, zero, max_leaves, size, height));
                }

                // the one quadrant node definition.
                if (y_lmid != x_mid) {
                    AABB<2> one = box;
                    one.mins[1] = y_split;
                    one.maxs[0] = x_split;

                    quads[1] = std::unique_ptr<Node>(new Node(y_lmid, x_mid, one, max_leaves, size, height));
                }

                // the two quadrant node definition.
                if (begin != y_lmid) {
                    AABB<2> two = box;
                    two.maxs[0] = x_split;
                    two.maxs[1] = y_split;

                    quads[2] = std::unique_ptr<Node>(new Node(begin, y_lmid, two, max_leaves, size, height));
                }

                // the three quadrant node definition.
                if (x_mid != y_rmid) {
                    AABB<2> three = box;
                    three.mins[0] = x_split;
                    three.maxs[1] = y_split;

                    quads[3] = std::unique_ptr<Node>(new Node(x_mid, y_rmid, three, max_leaves, size, height));
                }

                for (auto &quad : quads) {
                    if (quad) {
                        height++;
                        break;
                    }
                }
            }
        }
    };

    /** helper method for initial testing, locates the address of a point in the tree.
     *
     * @param subtree the root of the tree to search.
     * @param p the target point
     * @param closest a pointer to the leaf node of the quadrant in which the point lives.
     */
    void find_point(const std::unique_ptr<Node> &subtree, const Point<2> &p, std::unique_ptr<Node> *&closest) const {

        if (subtree) {
            std::unique_ptr<Node> *close = &subtree->quads[0];
            for (auto &quad : subtree->quads) {
                if (quad) {
                    if (distance(quad->bounds.closestInBox(p), p)
                        < distance((*close)->bounds.closestInBox(p), p)) {
                        close = &quad;
                    }
                }
            }
            find_point(*close, p, closest);
        }
    }

    std::unique_ptr<Node> root;        // The root node.
    int size = 0;                      // Helpful for tracking the size.
    int height = 0;

    /**
     *
     * @param subtree the root of the tree to search.
     * @param p the target point.
     * @param radius the distance around the target point to search for neighbors.
     * @param in_range a least of points encountered that are within the search radius.
     */
    void range_recursive(const std::unique_ptr<Node> &subtree,
                         const Point<2> &p,
                         const float radius,
                         std::vector<Point<2> > &in_range) const {

        if (subtree) {
            if (distance(subtree->bounds.closestInBox(p), p) < radius) {
                if (subtree->leaf) {
                    for (auto &point : subtree->node_points) {
                        if (distance(point, p) < radius) {
                            in_range.push_back(point);
                        }
                    }
                }
                for (auto &quad : subtree->quads) {
                    range_recursive(quad, p, radius, in_range);
                }
            }
        }
    }

    /** Recursive knn query, moves through the quadtree and updates a list of nearest
     * neighbors of size k as it works towards the target point and prunes subtrees.
     *
     * @param subtree the root node of the current tree.
     * @param p the target point.
     * @param k the number of nearest neighbors to locate.
     * @param nn a least of nearest neighbors encountered.
     */
    void knn_recursive(const std::unique_ptr<Node> &subtree,
                       const Point<2> &p,
                       int k,
                       std::vector<Point<2> > &nn) const {

        if (subtree) {
            if (subtree->leaf) {
                for (auto &point : subtree->node_points) {
                    if (nn.size() < k) {
                        nn.push_back(point);
                        std::push_heap(nn.begin(), nn.end(), DistanceComparator<2>(p));
                    } else if (distance(point, p) < distance(nn.front(), p)) {
                        std::pop_heap(nn.begin(), nn.end(), DistanceComparator<2>(p));
                        nn.pop_back();
                        nn.push_back(point);
                        std::push_heap(nn.begin(), nn.end(), DistanceComparator<2>(p));
                    }
                }
            } else {
                for (auto &quad : subtree->quads) {
                    if (quad) {
                        if (nn.size() < k || distance(quad->bounds.closestInBox(p), p) < distance(nn.front(), p)) {
                            knn_recursive(quad, p, k, nn);
                        }
                    }
                }
            }
        }
    }

};

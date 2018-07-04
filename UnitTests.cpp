/** Unit tests for 3 spacial-partitioning data structures, and a naive
 * implementation for comparative validation.
 *
 */
#define CATCH_CONFIG_MAIN

#include "catch.hpp"

#include <iostream>
#include <array>
#include <algorithm>
#include <cmath>

#include "KDTree.hpp"
#include "QuadTree.hpp"
#include "Point.hpp"
#include "Generators.hpp"
#include "Naive.hpp"
#include "VPTree.hpp"

/** Battery of tests to validate the naive methods before using them to test the other
 * spatial partitioning solutions and search structures.
 *
 * A collection of points in 3 dimensions is created, and the two search routines
 * are tested against the known outcomes.
 */
TEST_CASE("Validate Naive Methods") {

    int min = 0;
    int max = 4;
    std::vector<Point<3> > points = ValidationSet3D(min, max);

    // Validate that the generator creates the correct points.
    SECTION("Validate Generator") {
        auto size = static_cast<int>(pow(max, 3));
        // Make sure we get the requested number of points.
        REQUIRE(points.size() == size);
    }

    SECTION("Validate Naive KNN") {
        int k = 7;
        const auto target = points.back();
        std::vector<Point<3> > knns = naive_knn(points, target, k);
        // Check that we have found k nearest neighbors.
        REQUIRE(knns.size() == k);
        // Check that they are the correct neighbors.
        REQUIRE(knns.size() == k);
    }

    SECTION("Validate Naive RangeQuery") {
        float radius = 1.5f;
        Point<3> target = points.back();
        auto range_results = naive_range(points, target, radius).size();
        // Check that the upper corner (3,3,3) of a 4x4 cube (each point distance 1 from all other
        // points in the set) has 7 neighboring points within radius 1.5.
        REQUIRE(range_results == 7);
    }
}

/** Validates the KDTree data structure for a series of randomly generated point collections,
 * over a range of dimensions, and tests the results against the Naive implementation.
 *
 */
TEST_CASE("Validate KDTree Structure") {
    int min = 0;
    int max = 4;
    auto size = static_cast<int>(pow(max, 3));
    std::vector<Point<3> > points;
    points = ValidationSet3D(min, max);

    SECTION("Functional Validation") {
        int k = 5;
        Point<3> target = points[0];
        KDTree<3> kdtree(points);
        REQUIRE(kdtree.getSize() == size);
    }
}

TEST_CASE("Validate KDTree KNN") {

    SECTION("Implementation Check: Validate KNN") {
        int min = 0;
        int max = 20;
        int k = 2;
        std::vector<Point<1> > points = ValidationSet1D(min, max);
        Point<1> test = points[max / 2];

        auto _naive_knn = naive_knn(points, test, k);
    }

    int k = 30;

    UniformGenerator<3> gen(-100.0f, 100.0f);
    int training_size = 1000;

    TrialData<3> data = getTrialData<3>(1, training_size, gen);
    Point<3> target = data.training[0];

    auto _naive_knn = naive_knn(data.testing, target, k);
    KDTree<3> kdtree(data.testing);
    auto kdtree_knn = kdtree.knn(target, k);

    SECTION("Benchmark: Small uniform distribution") {

        // Basic tests to verify that the returned collection is the correct size (should be k).
        REQUIRE(_naive_knn.size() == kdtree_knn.size());

        // Prepare each collection for comparison by sorting.
        // The final comparison is performed by the CompareBy method starting with the 1st dimension.
        DistanceComparator<3> init_diff(target);
        std::sort(_naive_knn.begin(), _naive_knn.end(), init_diff);
        std::sort(kdtree_knn.begin(), kdtree_knn.end(), init_diff);

        print_points(kdtree_knn);
        print_points(_naive_knn);
        // Validate that the correct set of points has been found.
        for (int i = 0; i < k; i++) {
            REQUIRE(init_diff(_naive_knn[i], kdtree_knn[i]));
        }
    }
}

TEST_CASE("Validate KDTree RangeQuery") {

    SECTION("Implementation Check: Validate RangeQuery") {
        int min = 0;
        int max = 8;
        std::vector<Point<2> > points;
        points = ValidationSet2D(min, max);
        Point<2> target = points.back();
        float radius = 1.5;

        auto _naive_range = naive_range(points, target, radius);
        KDTree<2> kdtree(points);
        auto kdtree_range = kdtree.range_query(target, radius);

        // Basic test to verify that both range routines found the same number of points.
        REQUIRE(kdtree_range.size() == _naive_range.size());

        // Prepare each collection for comparison by sorting.
        CompareBy<0> init_diff;
        std::sort(_naive_range.begin(), _naive_range.end(), init_diff);
        std::sort(kdtree_range.begin(), kdtree_range.end(), init_diff);

        // Validate that the correct set of points has been found.
        for (int i = 0; i < _naive_range.size(); i++) {
            REQUIRE(_naive_range[i].point == kdtree_range[i].point);
        }
    }

    SECTION("Benchmark: Small gaussian distribution") {
        GaussianGenerator<3> gen(-100.0f, 100.0f);
        int training_size = 1000;

        TrialData<3> data = getTrialData<3>(1, training_size, gen);
        Point<3> target = data.training[0];
        float radius = 30;

        auto _naive_range = naive_range(data.testing, target, radius);
        KDTree<3> kdtree(data.testing);
        auto kdtree_range = kdtree.range_query(target, radius);

        // Basic test to verify that both range routines found the same number of points.
        REQUIRE(_naive_range.size() == kdtree_range.size());

        // Prepare each collection for comparison by sorting.
        CompareBy<0> init_diff;
        std::sort(_naive_range.begin(), _naive_range.end(), init_diff);
        std::sort(kdtree_range.begin(), kdtree_range.end(), init_diff);

        // Validate that the correct set of points has been found.
        for (int i = 0; i < _naive_range.size(); i++) {
            REQUIRE(_naive_range[i].point == kdtree_range[i].point);
        }
    }
}

/** Validates the QuadTree data structure for a series of randomly generated point collections,
 * over a range of dimensions, and tests the results against the Naive implementation.
 *
 */
TEST_CASE("Validate QuadTree Structure") {
    // create a 5x5 grid of points with a 1 point offset from the origin [0,0].
    int min = 0;
    int max = 4;
    int max_leaves = 4;
    auto size = static_cast<int>(pow(max, 2));
    std::vector<Point<2> > points;
    points = ValidationSet2D(min, max);

    // Validate size and height of QuadTree.
    SECTION("Functional Validation: Basic") {
        QuadTree quadtree1(points, max_leaves);
        REQUIRE(quadtree1.getSize() == size);
        REQUIRE(quadtree1.getHeight() == 1);
    }

    min = 0;
    max = 5;
    max_leaves = 4;
    size = static_cast<int>(pow(max, 2));
    points = ValidationSet2D(min, max);

    // Validate size and height of QuadTree.
    SECTION("Functional Validation: Middling") {
        QuadTree quadtree2(points, max_leaves);
        REQUIRE(quadtree2.getSize() == size);
        REQUIRE(quadtree2.getHeight() == 4);
    }

    min = 0;
    max = 100;
    max_leaves = 10;
    size = static_cast<int>(pow(max, 2));
    points = ValidationSet2D(min, max);

    // Validate size and height of QuadTree.
    SECTION("Functional Validation: Large") {
        QuadTree quadtree3(points, max_leaves);
        REQUIRE(quadtree3.getSize() == size);
        REQUIRE(quadtree3.getHeight() == 581);
    }
}

TEST_CASE("Validate QuadTree RangeQuery") {

    SECTION("Implementation Check: Validate RangeQuery") {
        int min = 0;
        int max = 4;
        std::vector<Point<2> > points;
        points = ValidationSet2D(min, max);
        Point<2> target = points.back();
        float radius = 1.5;
        int max_leaves = 4;

        auto _naive_range = naive_range(points, target, radius);
        QuadTree quadtree(points, max_leaves);
        auto quadtree_range = quadtree.range_query(target, radius);
        // Basic test to verify that both range routines found the same number of points.
        REQUIRE(_naive_range.size() == quadtree_range.size());

        // Prepare each collection for comparison by sorting.
        std::sort(_naive_range.begin(), _naive_range.end(), DistanceComparator<2>(target));
        std::sort(quadtree_range.begin(), quadtree_range.end(), DistanceComparator<2>(target));

        // Validate that the correct set of points has been found.
        for (int i = 0; i < _naive_range.size(); i++) {
            REQUIRE(_naive_range[i].point == quadtree_range[i].point);
        }
    }

    SECTION("Benchmark: Small gaussian distribution") {
        GaussianGenerator<2> gen(-100.0f, 100.0f);
        int training_size = 1000;

        TrialData<2> data = getTrialData<2>(1, training_size, gen);
        Point<2> target = data.training[0];
        float radius = 20;

        auto _naive_range = naive_range(data.testing, target, radius);
        int max_leaves = 4;
        QuadTree quadtree(data.testing, max_leaves);
        auto quadtree_range = quadtree.range_query(target, radius);

        // Basic test to verify that both range routines found the same number of points.
        REQUIRE(_naive_range.size() == quadtree_range.size());

        // Prepare each collection for comparison by sorting.
        DistanceComparator<2> init_diff(target);
        std::sort(_naive_range.begin(), _naive_range.end(), init_diff);
        std::sort(quadtree_range.begin(), quadtree_range.end(), init_diff);

        // Validate that the correct set of points has been found.
        for (int i = 0; i < _naive_range.size(); i++) {
            REQUIRE(_naive_range[i].point == quadtree_range[i].point);
        }
    }
}

TEST_CASE("Validate QuadTree KNN") {

    SECTION("Implementation Check: Validate KNN") {
        int min = 0;
        int max = 8;
        int k = 2;
        std::vector<Point<2> > points = ValidationSet2D(min, max);
        Point<2> test = points[max / 2];
        int max_leaves = 4;

        auto _naive_knn = naive_knn(points, test, k);
        QuadTree quadtree(points, max_leaves);
        auto quadtree_knn = quadtree.knn(test, k);

        REQUIRE(_naive_knn.size() == quadtree_knn.size());
    }

    int k = 3;

    UniformGenerator<2> gen(-100.0f, 100.0f);
    int training_size = 1000;

    TrialData<2> data = getTrialData<2>(1, training_size, gen);
    Point<2> target = data.training[0];

    auto _naive_knn = naive_knn(data.testing, target, k);
    int max_leaves = 4;
    QuadTree quadtree(data.testing, max_leaves);
    auto quadtree_knn = quadtree.knn(target, k);

    SECTION("Benchmark: Small uniform distribution") {

        // Basic tests to verify that the returned collection is the correct size (should be k).
        REQUIRE(_naive_knn.size() == quadtree_knn.size());

        // Prepare each collection for comparison by sorting.
        // The final comparison is performed by the CompareBy method starting with the 1st dimension.
        DistanceComparator<2> init_diff(target);
        std::sort(_naive_knn.begin(), _naive_knn.end(), init_diff);
        std::sort(quadtree_knn.begin(), quadtree_knn.end(), init_diff);

        // Validate that the correct set of points has been found.
        for (int i = 0; i < k; i++) {
            REQUIRE(distance(_naive_knn[i], quadtree_knn[i]) == 0);
        }
    }
}

/** Validates the VPTree data structure for a series of randomly generated point collections,
 * over a range of dimensions, and tests the results against the Naive implementation.
 *
 */
TEST_CASE("Validate VPTree Structure") {
    int min = 0;
    int max = 4;
    auto size = static_cast<int>(pow(max, 3));
    std::vector<Point<3> > points;
    points = ValidationSet3D(min, max);

    SECTION("Functional Validation") {
        Point<3> target = points[0];
        VPTree<3> vptree(points);
        REQUIRE(vptree.getSize() == size);
    }
}

TEST_CASE("Validate VPTree RangeQuery") {

    SECTION("Implementation Check: Validate RangeQuery") {
        int min = 0;
        int max = 3;
        std::vector<Point<2> > points;
        points = ValidationSet2D(min, max);
        Point<2> target = points.back();
        float radius = 1.5;

        auto _naive_range = naive_range(points, target, radius);
        VPTree<2> vptree(points);
        auto vptree_range = vptree.range_query(target, radius);

        // Basic test to verify that both range routines found the same number of points.
        REQUIRE(_naive_range.size() == vptree_range.size());

        // Prepare each collection for comparison by sorting.
        CompareBy<0> init_diff;
        std::sort(_naive_range.begin(), _naive_range.end(), init_diff);
        std::sort(vptree_range.begin(), vptree_range.end(), init_diff);

        // Validate that the correct set of points has been found.
        for (int i = 0; i < _naive_range.size(); i++) {
            REQUIRE(_naive_range[i].point == vptree_range[i].point);
        }
    }

    SECTION("Benchmark: Small gaussian distribution") {
        GaussianGenerator<3> gen(-100.0f, 100.0f);
        int training_size = 1000;

        TrialData<3> data = getTrialData<3>(1, training_size, gen);
        Point<3> target = data.training[0];
        float radius = 100;

        auto _naive_range = naive_range(data.testing, target, radius);
        VPTree<3> vptree(data.testing);
        auto vptree_range = vptree.range_query(target, radius);

        // Basic test to verify that both range routines found the same number of points.
        REQUIRE(_naive_range.size() == vptree_range.size());

        // Prepare each collection for comparison by sorting.
        CompareBy<0> init_diff;
        std::sort(_naive_range.begin(), _naive_range.end(), init_diff);
        std::sort(vptree_range.begin(), vptree_range.end(), init_diff);
        print_results(_naive_range, vptree_range);
        // Validate that the correct set of points has been found.
        for (int i = 0; i < _naive_range.size(); i++) {
            REQUIRE(_naive_range[i].point == vptree_range[i].point);
        }
    }
}

TEST_CASE("Validate VPTree KNN") {

    int k = 5;

    UniformGenerator<3> gen(-100.0f, 100.0f);
    int training_size = 1000;

    TrialData<3> data = getTrialData<3>(1, training_size, gen);
    Point<3> target = data.training[0];

    auto _naive_knn = naive_knn(data.testing, target, k);
    VPTree<3> vptree(data.testing);
    auto vptree_knn = vptree.knn(target, k);

    SECTION("Benchmark: Small uniform distribution") {

        // Basic tests to verify that the returned collection is the correct size (should be k).
        REQUIRE(_naive_knn.size() == vptree_knn.size());

        // Prepare each collection for comparison by sorting.
        // The final comparison is performed by the CompareBy method starting with the 1st dimension.
        CompareBy<0> init_diff;
        std::sort(_naive_knn.begin(), _naive_knn.end(), init_diff);
        std::sort(vptree_knn.begin(), vptree_knn.end(), init_diff);

        print_results(_naive_knn, vptree_knn);
        // Validate that the correct set of points has been found.
        for (int i = 0; i < k; i++) {
            REQUIRE(_naive_knn[i].point == vptree_knn[i].point);
        }
    }
}
/** Benchmark methods for generating data comparative analysis data for the KNN methods of
 * each data structure.
 *
 * There is an enormous amount of repeated code here that could be cleaned up with a better
 * program structure (specifically creating a generator class, so we could put the generators
 * in a vector and iterate through them), and creating a search container type to extend from for
 * each of the data structures (again, easier to iterate over without all the extra, repeated
 * code).
 *
 * If we had more time I would certainly refactor this mess.
 *
 */

#pragma once

#include "catch.hpp"

#include <iostream>
#include <fstream>
#include <array>
#include <algorithm>
#include <cmath>
#include <string>

#include "KDTree.hpp"
#include "Point.hpp"
#include "Generators.hpp"
#include "Naive.hpp"
#include "Stopwatch.hpp"
#include "QuadTree.hpp"
#include "VPTree.hpp"

// For lack of a better solution, the global dimension for all
// structure templates is defined here.
const int d = 21;

struct test_conditions {
    test_conditions() = default;

    int replicates = 10;        // number of replicates to average for each test condition.
    int n = 10;                 // number of queries to perform for each test step.

    int k_interval = 2;         // interval increment size.
    int k_max = 30;             // maximum number of neighbors to search for in KNN.

    int points_init = 100;      // interval to increment the point collection size.
    int points_max = 26000;     // maximum size of the point collection to test.
    int points_fold = 2;

    float min = -1000;          // minimum value of a point in dimension n.
    float max = 1000;           // max value of a point in dimension n.
};

/** helper method for exporting data to a csv file. the dimension of the tests is used as part
 * of the naming convention for use when interrogating the file.
 *
 */
void export_to_csv(const std::vector<std::string> &results, const std::string &header) {
    std::ofstream file;
    file.open("results_" + std::to_string(d) + "D.csv");
    for (int i = 0; i < results.size(); i++) {
        file << results[i];
    }
}

/** performs a benchmark for the kdtree spatial partitioning data structure,
 * using the parameters defined in the test_conditions struct.
 *
 */
void benchmark_kdtree(std::vector<std::string> &results) {
    std::string type = "KDTree";
    const test_conditions c;
    double average_search_time = 0;

    Stopwatch sw;

    // create the generators of the appropriate dimension.
    GaussianGenerator<d> gen_gaussian(c.min, c.max);
    UniformGenerator<d> gen_uniform(c.min, c.max);
    // iterate through the range of point collection sizes, with gaussian.
    for (int points = c.points_init; points < c.points_max; points *= c.points_fold) {

        auto data = getTrialData<d>(c.n, points, gen_gaussian);
        KDTree<d> kdtree(data.testing);

        for (int k = c.k_interval; k < c.k_max; k += c.k_interval) {
            average_search_time = 0;
            for (int n = 0; n < c.n; n++) {
                auto target = data.training[n];
                for (int reps = 0; reps < c.replicates; reps++) {

                    sw.start();
                    kdtree.knn(target, k);
                    average_search_time += sw.stop();
                }
            }
            average_search_time /= (c.replicates * c.n);
            results.emplace_back(
                    std::to_string(points) + ","
                    + "gaussian,"
                    + std::to_string(k) + ","
                    + std::to_string(average_search_time) + ","
                    + type + ","
                    + std::to_string(d) + "\n");
        }
    }

    // iterate through the range of point collection sizes.
    for (int points = c.points_init; points < c.points_max; points *= c.points_fold) {

        auto data = getTrialData<d>(c.n, points, gen_uniform);
        KDTree<d> kdtree(data.testing);

        for (int k = c.k_interval; k < c.k_max; k += c.k_interval) {
            average_search_time = 0;
            for (int n = 0; n < c.n; n++) {
                auto target = data.training[n];
                for (int reps = 0; reps < c.replicates; reps++) {

                    sw.start();
                    kdtree.knn(target, k);
                    average_search_time += sw.stop();
                }

            }
            average_search_time /= (c.replicates * c.n);
            results.emplace_back(
                    std::to_string(points) + ","
                    + "uniform,"
                    + std::to_string(k) + ","
                    + std::to_string(average_search_time) + ","
                    + type + ","
                    + std::to_string(d) + "\n");
        }
    }
}

/** performs benchmark of brute force, naive solutions to knn and range querries on a collection
 * of points.
 *
 */
void benchmark_naive(std::vector<std::string> &results) {
    std::string type = "naive";
    const test_conditions c;
    double average_search_time = 0;

    Stopwatch sw;

    // create the generators of the appropriate dimension.
    GaussianGenerator<d> gen_gaussian(c.min, c.max);
    UniformGenerator<d> gen_uniform(c.min, c.max);

    // iterate through the range of point collection sizes, with gaussian generator.
    for (int points = c.points_init; points < c.points_max; points *= c.points_fold) {

        auto data = getTrialData<d>(c.n, points, gen_gaussian);

        for (int k = c.k_interval; k < c.k_max; k += c.k_interval) {
            average_search_time = 0;
            for (int n = 0; n < c.n; n++) {
                auto target = data.training[n];
                for (int reps = 0; reps < c.replicates; reps++) {

                    sw.start();
                    naive_knn(data.testing, target, k);
                    average_search_time += sw.stop();
                }

            }
            average_search_time /= (c.replicates * c.n);
            results.emplace_back(
                    std::to_string(points) + ","
                    + "gaussian,"
                    + std::to_string(k) + ","
                    + std::to_string(average_search_time) + ","
                    + type + ","
                    + std::to_string(d) + "\n");
        }
    }

    // iterate through the range of point collection sizes, with a uniform generator.
    for (int points = c.points_init; points < c.points_max; points *= c.points_fold) {

        auto data = getTrialData<d>(c.n, points, gen_uniform);

        for (int k = c.k_interval; k < c.k_max; k += c.k_interval) {
            average_search_time = 0;
            for (int n = 0; n < c.n; n++) {
                auto target = data.training[n];
                for (int reps = 0; reps < c.replicates; reps++) {

                    sw.start();
                    naive_knn(data.testing, target, k);
                    average_search_time += sw.stop();
                }

            }
            average_search_time /= (c.replicates * c.n);
            results.emplace_back(
                    std::to_string(points) + ","
                    + "uniform,"
                    + std::to_string(k) + ","
                    + std::to_string(average_search_time) + ","
                    + type + ","
                    + std::to_string(d) + "\n");
        }

    }
}

/** performs a benchmark for the quadtree spatial partitioning data structure,
 * using the parameters defined in the test_conditions struct.
 *
 * Test restricted to 2 Dimensions.
 *
 */
void benchmark_quadtree(std::vector<std::string> &results) {
    std::string type = "QuadTree";
    const test_conditions c;
    double average_search_time = 0;

    Stopwatch sw;

    // create the generators of the appropriate dimension.
    GaussianGenerator<2> gen_gaussian(c.min, c.max);
    UniformGenerator<2> gen_uniform(c.min, c.max);
    // iterate through the range of point collection sizes, with gaussian.
    int max_leaves = 4;
    for (int points = c.points_init; points < c.points_max; points *= c.points_fold) {

        auto data = getTrialData<2>(c.n, points, gen_gaussian);

        QuadTree kdtree(data.testing, max_leaves);

        for (int k = c.k_interval; k < c.k_max; k += c.k_interval) {
            average_search_time = 0;
            for (int n = 0; n < c.n; n++) {
                auto target = data.training[n];
                for (int reps = 0; reps < c.replicates; reps++) {

                    sw.start();
                    kdtree.knn(target, k);
                    average_search_time += sw.stop();
                }
            }
            average_search_time /= (c.replicates * c.n);
            results.emplace_back(
                    std::to_string(points) + ","
                    + "gaussian,"
                    + std::to_string(k) + ","
                    + std::to_string(average_search_time) + ","
                    + type + ","
                    + std::to_string(d) + "\n");
        }
    }

    // iterate through the range of point collection sizes.
    for (int points = c.points_init; points < c.points_max; points *= c.points_fold) {

        auto data = getTrialData<2>(c.n, points, gen_uniform);
        QuadTree kdtree(data.testing, max_leaves);

        for (int k = c.k_interval; k < c.k_max; k += c.k_interval) {
            average_search_time = 0;
            for (int n = 0; n < c.n; n++) {
                auto target = data.training[n];
                for (int reps = 0; reps < c.replicates; reps++) {

                    sw.start();
                    kdtree.knn(target, k);
                    average_search_time += sw.stop();
                }

            }
            average_search_time /= (c.replicates * c.n);
            results.emplace_back(
                    std::to_string(points) + ","
                    + "uniform,"
                    + std::to_string(k) + ","
                    + std::to_string(average_search_time) + ","
                    + type + ","
                    + std::to_string(d) + "\n");
        }
    }
}

/** performs a benchmark for the kdtree spatial partitioning data structure,
 * using the parameters defined in the test_conditions struct.
 *
 */
void benchmark_vptree(std::vector<std::string> &results) {
    std::string type = "VPTree";
    const test_conditions c;
    double average_search_time = 0;

    Stopwatch sw;

    // create the generators of the appropriate dimension.
    GaussianGenerator<d> gen_gaussian(c.min, c.max);
    UniformGenerator<d> gen_uniform(c.min, c.max);
    // iterate through the range of point collection sizes, with gaussian.
    for (int points = c.points_init; points < c.points_max; points *= c.points_fold) {

        auto data = getTrialData<d>(c.n, points, gen_gaussian);
        VPTree<d> vptree(data.testing);

        for (int k = c.k_interval; k < c.k_max; k += c.k_interval) {
            average_search_time = 0;
            for (int n = 0; n < c.n; n++) {
                auto target = data.training[n];
                for (int reps = 0; reps < c.replicates; reps++) {

                    sw.start();
                    vptree.knn(target, k);
                    average_search_time += sw.stop();
                }
            }
            average_search_time /= (c.replicates * c.n);
            results.emplace_back(
                    std::to_string(points) + ","
                    + "gaussian,"
                    + std::to_string(k) + ","
                    + std::to_string(average_search_time) + ","
                    + type + ","
                    + std::to_string(d) + "\n");
        }
    }

    // iterate through the range of point collection sizes.
    for (int points = c.points_init; points < c.points_max; points *= c.points_fold) {

        auto data = getTrialData<d>(c.n, points, gen_uniform);
        VPTree<d> vptree(data.testing);

        for (int k = c.k_interval; k < c.k_max; k += c.k_interval) {
            average_search_time = 0;
            for (int n = 0; n < c.n; n++) {
                auto target = data.training[n];
                for (int reps = 0; reps < c.replicates; reps++) {

                    sw.start();
                    vptree.knn(target, k);
                    average_search_time += sw.stop();
                }
            }
            average_search_time /= (c.replicates * c.n);
            results.emplace_back(
                    std::to_string(points) + ","
                    + "uniform,"
                    + std::to_string(k) + ","
                    + std::to_string(average_search_time) + ","
                    + type + ","
                    + std::to_string(d) + "\n");
        }
    }
}

void quadtree_max_leaves_exploration() {

    std::vector<std::string> results;
    std::string header = "points,distribution,k,avg_knn_time,structure,max_leaves\n";
    results.push_back(header);
    std::string type = "QuadTree";
    const test_conditions c;
    double average_search_time = 0;

    Stopwatch sw;

    // create the generators of the appropriate dimension.
    GaussianGenerator<2> gen_gaussian(c.min, c.max);
    UniformGenerator<2> gen_uniform(c.min, c.max);
    // iterate through the range of point collection sizes, with gaussian.
    int points = 1000;
    int k = 30;
    for (int m = 1; m < 10; m++) {

        auto data = getTrialData<2>(c.n, points, gen_gaussian);

        QuadTree quadtree(data.testing, m);

        average_search_time = 0;
        for (int n = 0; n < c.n; n++) {
            auto target = data.training[n];
            for (int reps = 0; reps < c.replicates; reps++) {

                sw.start();
                quadtree.knn(target, k);
                average_search_time += sw.stop();
            }
        }
        average_search_time /= (c.replicates * c.n);
        results.emplace_back(
                std::to_string(points) + ","
                + "gaussian,"
                + std::to_string(k) + ","
                + std::to_string(average_search_time) + ","
                + type + ","
                + std::to_string(m) + "\n");

    }

    // iterate through the range of point collection sizes.
    for (int m = 1; m < 10; m++) {

        auto data = getTrialData<2>(c.n, points, gen_uniform);
        QuadTree quadtree(data.testing, m);


        average_search_time = 0;
        for (int n = 0; n < c.n; n++) {
            auto target = data.training[n];
            for (int reps = 0; reps < c.replicates; reps++) {

                sw.start();
                quadtree.knn(target, k);
                average_search_time += sw.stop();
            }

        }
        average_search_time /= (c.replicates * c.n);
        results.emplace_back(
                std::to_string(points) + ","
                + "uniform,"
                + std::to_string(k) + ","
                + std::to_string(average_search_time) + ","
                + type + ","
                + std::to_string(m) + "\n");

    }

    export_to_csv(results, "Exploration of the number of max_leaves");

}

/** performs a benchmark on all spatial partitioning data structures based on the test criterion
 * in the test_condition struct.
 *
 * @return
 */
static std::vector<std::string> benchmark_all() {
    Stopwatch clock;

    std::vector<std::string> results;
    std::string header = "points,distribution,k,avg_knn_time,structure,dimensions\n";
    results.push_back(header);
    clock.start();

    benchmark_kdtree(results);
    if (d == 2) {
        benchmark_naive(results);
        benchmark_quadtree(results);
    }
    benchmark_vptree(results);
    double mark = clock.stop();

    export_to_csv(results, "");

    std::cout << "Benchmarks completed in ";
    std::cout << std::to_string(mark) + " s\n";

    return results;
}
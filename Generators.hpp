/** Methods for generating collections of points with specific distributions, within
 * the target dimension space.
 *
 */

#pragma once

#include "Point.hpp"
#include <random>

/** Generates a collection of points from a uniform distribution.
 *
 * @tparam Dimension the current dimension space.
 */
template<int Dimension>
struct UniformGenerator {

    UniformGenerator(float min_, float max_)
            : min(min_), max(max_), rd{}, gen(rd()), dis(min_, max_) {}

    Point<Dimension> generatePoint() {

        std::array<float, Dimension> data;
        for (int i = 0; i < Dimension; ++i) {
            data[i] = dis(gen);
        }
        return Point<Dimension>{data};
    }

private:
    float min, max;
    std::random_device rd;          // Used to obtain a seed for the random number engine.
    std::mt19937 gen;               // Standard mersenne_twister_engine seeded with rd().
    std::uniform_real_distribution<> dis;

};

/** Generates a collection of points with a gaussian (mean clustered) distribution.
 * The standard deviation governs the spread of the points (away from the mean).
 * @tparam Dimension the current dimension space.
 */

template<int Dimension>
struct GaussianGenerator {

    GaussianGenerator(float mean_, float stdDev_)
            : mean(mean_), stdDev(stdDev_), rd{}, gen(rd()), dis(mean_, stdDev_) {}

    Point<Dimension> generatePoint() {

        std::array<float, Dimension> data;
        for (int i = 0; i < Dimension; ++i) {
            data[i] = dis(gen);
        }
        return Point<Dimension>{data};
    }

private:
    float mean, stdDev;
    std::random_device rd;          // Used to obtain a seed for the random number engine.
    std::mt19937 gen;               // Standard mersenne_twister_engine seeded with rd().
    std::normal_distribution<> dis;

};

/** A testing-oriented structure, providing a pair of point collections designed for
 * the common training/testing methodology.
 *
 * @tparam Dimension the current dimension space.
 */
template<int Dimension>
struct TrialData {
    std::vector<Point<Dimension> > training, testing;
};

template<int Dimension, typename Generator>
TrialData<Dimension> getTrialData(int trainingSize, int testingSize, Generator &gen) {
    TrialData<Dimension> ret;
    for (int i = 0; i < trainingSize; ++i) {
        ret.training.push_back(gen.generatePoint());
    }
    for (int i = 0; i < testingSize; ++i) {
        ret.testing.push_back(gen.generatePoint());
    }
    return ret;
}

/** Generates an evenly, ordered distribution of 3D points for testing.
 *
 * Must be used with Dimension <3>.
 *
 * @tparam Dimension the current dimension space.
 */

std::vector<Point<3> > ValidationSet3D(const int min, const int max) {

    std::array<float, 3> data;
    std::vector<Point<3> > points;

    for (int i = min; i < max; i++) {
        for (int j = min; j < max; j++) {
            for (int k = min; k < max; k++) {

                data[0] = i;
                data[1] = j;
                data[2] = k;
                points.push_back(Point<3>{data});
            }
        }
    }
    return points;
}
/** Generates an evenly, ordered distribution of 3D points for validation.
 *
 * Must be used with Dimension <3>.
 *
 * @tparam Dimension the current dimension space.
 */

std::vector<Point<2> > ValidationSet2D(const int min, const int max) {

    std::array<float, 2> data;
    std::vector<Point<2> > points;

    for (int i = min; i < max; i++) {
        for (int j = min; j < max; j++) {
            data[0] = i;
            data[1] = j;
            points.push_back(Point<2>{data});
        }
    }
    return points;
}

/** Generates an evenly, ordered distribution of 3D points for validation.
 *
 * Must be used with Dimension <3>.
 *
 * @tparam Dimension the current dimension space.
 */
std::vector<Point<1> > ValidationSet1D(const int min, const int max) {

    std::array<float, 1> data;
    std::vector<Point<1> > points;

    for (int i = min; i < max; i++) {

        data[0] = i;
        points.push_back(Point<1>{data});
    }
    return points;
}
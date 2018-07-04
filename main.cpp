#include <iostream>
#include "KDTree.hpp"
#include "Point.hpp"
#include "Generators.hpp"
#include "Benchmark.hpp"
#include "Spinner.hpp"
#include "QuadTree.hpp"
#include "VPTree.hpp"

int main() {
    spinners::Spinner spinner;
    spinner.setText("Running Spatial Partitioning Benchmarks");
    spinner.setInterval(100);
    spinner.setSymbols("dots");

    spinner.start();
    benchmark_all();
    spinner.stop();

}
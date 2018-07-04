#pragma once

#include "Point.hpp"
#include <vector>

template<int Dimension> //The dimension of the points.  This can be used like any other constant within.
class BucketKNN{


public:

  BucketKNN(const std::vector<Point<Dimension> >& points, int divisions_)
  {
	
  }

    std::vector<Point<Dimension> > rangeQuery(const Point<Dimension> &p, float radius) const {

        return {};
    }


  std::vector<Point<Dimension> > KNN(const Point<Dimension>& p, int k) const{

	assert(numPoints >= k);

	return {};
	
  }

  


private:

  
  
};

# Notes On Implementation

## Iterator stuff
```
Iter start, stop; //If I have 2 iterators

//I can subtract them to find how many elements are in between
int size = stop - start; 

//I can get add a number to get an iterator to some other element
Iter middle = start + size/2;

//I can derefence it to get the value
Point middleValue = *middle;

//find first quartile and partition
CompareBy<0> compareByX; //See the Points.hpp file

//Find the first quartile and partition accordingly
std::nth_element(start, start + size/4, end, compareByX);
Point quartileValue = *(start + size/4);
```

## Looping over Buckets

Pseudocode for looking at all the buckets when we don't know the number of dimensions in advance (so we can't just use nested for loops)

```
int[] minCoords = findBucket(point - radius)
int[] maxCoords = findBucket(point + radius)

//We want to do this
for each bucket from minCoords to maxCoords:
	process(bucket)


//We can do it with something like this
for( coords = minBucket; //start at the beginning
     coords != nextBucket(maxCoords); //stop once we go past the end
     coords = nextBucket(coords, minCoords, maxCoords)  
        //advance to the next set of coordinates
    ){
  process(bucket(coords))
}

//pseudocode for next bucket
int[] nextBucket(int[] current, int[] minCoords, int[] maxCoords){
  current[lastDimension] ++; //increment the last dimension
  for( i = lastDimension; i > 0; —i){
    //if we need to "carry"
    if(current[i] > maxCoords[i]){
      //reset this dimension
      current[i] = minCoords[i];
      //and add to the next "digit"
      current[i -1]++;
    } else {
      //no more carries... we're done here
      break;
   }  
  }
}
```
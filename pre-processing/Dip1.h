//============================================================================
// Name        : Dip1.h
// Author      : Ronny Haensch
// Version     : 2.0
// Copyright   : -
// Description : header file for first DIP assignment
//============================================================================

#include <iostream>
#include <opencv2/opencv.hpp>

#define max(a,b) (a>=b?a:b)
#define min(a,b) (a<=b?a:b)

using namespace std;
using namespace cv;

class Dip1{

	public:
		// constructor
		Dip1(void){};
		// destructor
		~Dip1(void){};
		
		// processing routine
		void run(string);
		// testing routine
		void test(string);

	private:
		// function that performs some kind of (simple) image processing
		// --> edit ONLY this function!
		Mat doSomethingThatMyTutorIsGonnaLike(Mat&);

		// test function
		void test_doSomethingThatMyTutorIsGonnaLike(Mat&, Mat&);
};

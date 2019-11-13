// Generates a gaussian filter kernel of given size
/*
kSize:     kernel size (used to calculate standard deviation)
return:    the generated filter kernel
*/
Mat Dip3::createGaussianKernel(int kSize){

  //create symmetric kernel
	Mat Kernel = Mat(kSize, kSize, CV_32FC1);
	double sigma = kSize/kSize;

	double mean = kSize/2;

	double sum = 0.0; // Accumulate kernel values
	for (int x = 0; x < kSize; ++x)
	    for (int y = 0; y < kSize; ++y) {
	        Kernel.at<float>(x,y)= exp( -0.5 * (pow((x-mean)/sigma, 2.0) + pow((y-mean)/sigma,2.0)) )
	                         / (2 * M_PI * sigma * sigma);

	        // Accumulate kernel values
	        sum += Kernel.at<float>(x,y);
	    }

	// Normalize the kernel
	for (int x = 0; x < kSize; ++x)
	    for (int y = 0; y < kSize; ++y)
	    	Kernel.at<float>(x,y) /= sum;

	

  return Kernel;
}


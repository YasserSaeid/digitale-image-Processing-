//============================================================================
// Name    : Dip3.cpp
// Author   : Ronny Haensch
// Version    : 2.0
// Copyright   : -
// Description : 
//============================================================================

#include "Dip3.h"

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



// Performes a circular shift in (dx,dy) direction
/*
in       input matrix
dx       shift in x-direction
dy       shift in y-direction
return   circular shifted matrix
*/
Mat Dip3::circShift(Mat& in, int dx, int dy){

	Mat res = in.clone();

	int yy, xx;

	for (int y = 0; y < res.rows; y++)
	{
		//shift in y direction
		yy = (y + dy) % res.rows;

		//row border
		if(yy<0)
			yy = yy + res.rows;

		for (int x = 0; x < res.cols; x++)
		{
			//shift in x direction
			xx = (x + dx) % res.cols;

			//col border
			if(xx < 0)
				xx = xx + res.cols;

			//shift values
			res.at<float>(yy*res.cols + xx) = in.at<float>(y * res.cols + x);
		}

	}


   // TO DO !!!

   return res;
}

//Performes a convolution by multiplication in frequency domain
/*
in       input image
kernel   filter kernel
return   output image
*/
Mat Dip3::frequencyConvolution(Mat& in, Mat& kernel){
	// copy of the kernel in a matrix with in size
	  int in_row = in.rows;
	  int in_col = in.cols;

	  int k_row = kernel.rows;
	  int k_col = kernel.cols;

	  Mat new_kernel =  Mat::zeros(in_row, in_col, CV_32FC1);


	  for (int i = 0 ; i < k_row ; i ++)
	  {
	    for (int j = 0 ; j < k_col ; j ++)
	    {
	      new_kernel.at<float>(i,j) = kernel.at<float>(i,j);
	    }
	  }

	  Mat shift_kernel = circShift(new_kernel, -k_row/2, -k_col/2);

	  //Forward transform:
	  Mat F_kernel;
	  dft(shift_kernel, F_kernel, 0 );

	  Mat F_input;
	  dft(in, F_input, 0 );

	  //Spectrum multiplication
	  Mat Convol;
	  mulSpectrums(F_input, F_kernel, Convol, 0 );

	  //Inverse transform
	  Mat output;
	  dft(Convol, output, DFT_INVERSE + DFT_SCALE);


	   return output;
}

// Performs UnSharp Masking to enhance fine image structures
/*
in       input image
type     integer defining how convolution for smoothing operation is done
         0 <==> spatial domain; 1 <==> frequency domain; 2 <==> seperable filter; 3 <==> integral image
size     size of used smoothing kernel
thresh   minimal intensity difference to perform operation
scale    scaling of edge enhancement
return   enhanced image
*/
Mat Dip3::usm(Mat& in, int type, int size, double thresh, double scale){

   // some temporary images 
   Mat tmp(in.rows, in.cols, CV_32FC1);
   Mat diff(in.rows, in.cols, CV_32FC1);
   Mat finalImage(in.rows, in.cols, CV_32FC1);
   

   // calculate edge enhancement

   // 1: smooth original image
   //    save result in tmp for subsequent usage
   switch(type){
      case 0:
         tmp = mySmooth(in, size, 0);
         break;
      case 1:
         tmp = mySmooth(in, size, 1);
         break;
      case 2: 
	tmp = mySmooth(in, size, 2);
        break;
      case 3: 
	tmp = mySmooth(in, size, 3);
        break;
      default:
         GaussianBlur(in, tmp, Size(floor(size/2)*2+1, floor(size/2)*2+1), size/5., size/5.);
   }

   Mat E(in.size(), in.type());
   diff.convertTo(E, CV_32FC1);

   Mat C( in.size(), in.type() );
   tmp.convertTo(C, CV_32FC1);

   subtract(in, tmp, diff);


 	//Mat semiFinal = in + diff;
    Mat F(in.size(), in.type());
    finalImage.convertTo(F, CV_32FC1);


    threshold(diff, diff, thresh, 255, THRESH_TOZERO);

    add(in,(scale*diff),finalImage);


  cout << "num pixels: " << in.total() << endl;

      //return finalImage;
    return finalImage;

}

// convolution in spatial domain
/*
src:    input image
kernel:  filter kernel
return:  convolution result
*/
Mat Dip3::spatialConvolution(Mat& src, Mat& kernel){
	int kSize = kernel.rows;
		// we assume kernel is a square matrix
		int srcRow = src.rows;
		int srcCol = src.cols;

	  Mat output =  Mat::zeros(srcRow,srcCol, CV_32FC1);


	  // flip the kernel

	  Mat kernel_flip =  Mat::zeros(kSize,kSize, CV_32FC1);

	  for (int k = 0 ; k < kSize ; k ++)
	  {
	    for (int l = 0 ; l < kSize ; l ++)
	    {
	      kernel_flip.at<float>(k, l) = kernel.at<float>(kSize - 1 - k, kSize - 1 - l);
	    }
	  }



		for (int i = 0; i < srcRow; i++)
		{
			for (int j = 0; j < srcCol; j++)
			{
	      float res = 0; //  will contain the sum of all the convoluted terms.

				for (int k = 0; k < kSize; k++)
				{
					int k_src = i - (kSize - 1) / 2 + k;
					if (k_src < 0)
					{
						k_src = 0;
					}
					else if (k_src >= srcRow)
					{
						k_src = srcRow - 1;
					}
					for (int l = 0; l < kSize; l++)
					{
						int l_src = j + l - (kSize - 1) / 2;
						if (l_src < 0)
						{
							l_src = 0;
						}
						else if (l_src >= srcCol)
						{
							l_src = srcCol - 1;
						}

						res = res + src.at<float>(k_src,l_src) * kernel_flip.at<float>(k, l);
					}
				}

				output.at<float>(i,j) = res;


			}

		}

		return output;
}

// convolution in spatial domain by seperable filters
/*
src:    input image
size     size of filter kernel
return:  convolution result
*/
Mat Dip3::seperableFilter(Mat& src, int size){

   // optional

   return src;

}

// convolution in spatial domain by integral images
/*
src:    input image
size     size of filter kernel
return:  convolution result
*/
Mat Dip3::satFilter(Mat& src, int size){

   // optional

   return src;

}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

// function calls processing function
/*
in       input image
type     integer defining how convolution for smoothing operation is done
         0 <==> spatial domain; 1 <==> frequency domain
size     size of used smoothing kernel
thresh   minimal intensity difference to perform operation
scale    scaling of edge enhancement
return   enhanced image
*/
Mat Dip3::run(Mat& in, int smoothType, int size, double thresh, double scale){

   return usm(in, smoothType, size, thresh, scale);

}


// Performes smoothing operation by convolution
/*
in       input image
size     size of filter kernel
type     how is smoothing performed?
return   smoothed image
*/
Mat Dip3::mySmooth(Mat& in, int size, int type){

   // create filter kernel
   Mat kernel = createGaussianKernel(size);
 
   // perform convoltion
   switch(type){
     case 0: return spatialConvolution(in, kernel);	// 2D spatial convolution
     case 1: return frequencyConvolution(in, kernel);	// 2D convolution via multiplication in frequency domain
     case 2: return seperableFilter(in, size);	// seperable filter
     case 3: return satFilter(in, size);		// integral image
     default: return frequencyConvolution(in, kernel);
   }
}

// function calls some basic testing routines to test individual functions for correctness
void Dip3::test(void){

   test_createGaussianKernel();
   test_circShift();
   test_frequencyConvolution();
   cout << "Press enter to continue"  << endl;
   cin.get();

}

void Dip3::test_createGaussianKernel(void){

   Mat k = createGaussianKernel(11);
   
   if ( abs(sum(k).val[0] - 1) > 0.0001){
      cout << "ERROR: Dip3::createGaussianKernel(): Sum of all kernel elements is not one!" << endl;
      return;
   }
   if (sum(k >= k.at<float>(5,5)).val[0]/255 != 1){
      cout << "ERROR: Dip3::createGaussianKernel(): Seems like kernel is not centered!" << endl;
      return;
   }
   cout << "Message: Dip3::createGaussianKernel() seems to be correct" << endl;
}

void Dip3::test_circShift(void){
   
   Mat in = Mat::zeros(3,3,CV_32FC1);
   in.at<float>(0,0) = 1;
   in.at<float>(0,1) = 2;
   in.at<float>(1,0) = 3;
   in.at<float>(1,1) = 4;
   Mat ref = Mat::zeros(3,3,CV_32FC1);
   ref.at<float>(0,0) = 4;
   ref.at<float>(0,2) = 3;
   ref.at<float>(2,0) = 2;
   ref.at<float>(2,2) = 1;
   
   if (sum((circShift(in, -1, -1) == ref)).val[0]/255 != 9){
      cout << "ERROR: Dip3::circShift(): Result of circshift seems to be wrong!" << endl;
      return;
   }
   cout << "Message: Dip3::circShift() seems to be correct" << endl;
}

void Dip3::test_frequencyConvolution(void){
   
   Mat input = Mat::ones(9,9, CV_32FC1);
   input.at<float>(4,4) = 255;
   Mat kernel = Mat(3,3, CV_32FC1, 1./9.);

   Mat output = frequencyConvolution(input, kernel);
   
   if ( (sum(output < 0).val[0] > 0) or (sum(output > 255).val[0] > 0) ){
      cout << "ERROR: Dip3::frequencyConvolution(): Convolution result contains too large/small values!" << endl;
      return;
   }
   float ref[9][9] = {{0, 0, 0, 0, 0, 0, 0, 0, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 1, 1, (8+255)/9., (8+255)/9., (8+255)/9., 1, 1, 0},
                      {0, 1, 1, (8+255)/9., (8+255)/9., (8+255)/9., 1, 1, 0},
                      {0, 1, 1, (8+255)/9., (8+255)/9., (8+255)/9., 1, 1, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 0, 0, 0, 0, 0, 0, 0, 0}};
   for(int y=1; y<8; y++){
      for(int x=1; x<8; x++){
         if (abs(output.at<float>(y,x) - ref[y][x]) > 0.0001){
            cout << "ERROR: Dip3::frequencyConvolution(): Convolution result contains wrong values!" << endl;
            return;
         }
      }
   }
   cout << "Message: Dip3::frequencyConvolution() seems to be correct" << endl;
}

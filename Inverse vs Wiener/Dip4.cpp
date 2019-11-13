//============================================================================
// Name        : Dip4.cpp
// Author      : Ronny Haensch
// Version     : 2.0
// Copyright   : -
// Description : 
//============================================================================

#include "Dip4.h"

// Performes a circular shift in (dx,dy) direction
/*
in       :  input matrix
dx       :  shift in x-direction
dy       :  shift in y-direction
return   :  circular shifted matrix
*/


Mat Dip4::circShift(Mat& in, int dx, int dy){
int row = in.rows;
  int col = in.cols;

  Mat output = Mat::zeros(row, col, CV_32FC1);

  for (int i = 0; i < row ; i ++)
  {
    for (int j = 0 ; j < col ; j ++)
    {
      float x = i + dx;
      float y = j + dy; 

      if (x < 0)
      {
        x = x + row;
      }
      if (x >= row) // as dx = -in.row/2 this case won't probably happen...
      {
        x = x - row ; 
      }
      if (y < 0)
      {
        y = y + col;
      }
      if (x >= col)
      {
        x = x - col ; 
      }

      output.at<float>(x,y) = in.at<float>(i,j);

    }
  }

   return output;
}




// Function applies inverse filter to restorate a degraded image
/*
degraded :  degraded input image
filter   :  filter which caused degradation
return   :  restorated output image
*/
Mat Dip4::inverseFilter(Mat& degraded, Mat& filter){


  // Creation of a shifted filter with degraded image size

  int d_row = degraded.rows; 
  int d_col = degraded.cols; 

  Mat filter_resize = Mat::zeros(d_row, d_col, CV_32FC1); 

  for (int i = 0 ; i < filter.rows ; i ++)
  {
    for (int j = 0 ; j < filter.cols ; j++)
    {
      filter_resize.at<float>(i,j) = filter.at<float>(i,j); 
    }
  }


 filter_resize = circShift(filter_resize, -filter.rows/2, -filter.cols/2);


  // Fourier transform of the degraded image and of the (resized and shifted) filter

  Mat degraded_ft; 
  Mat filter_ft; 

  dft(degraded, degraded_ft, DFT_COMPLEX_OUTPUT);
  dft(filter_resize, filter_ft,  DFT_COMPLEX_OUTPUT);

  // Separation of real and imaginary part of the filter in frequency domain

  Mat complexChannel[2];
  split(filter_ft, complexChannel);

  Mat Re = complexChannel[0];
  Mat Im = complexChannel[1]; 

  // calculation of the threshold

  float epsilon = 0.05; 

  int row = filter_ft.rows;
  int col = filter_ft.cols;

  float max = 0; 

  for (int i = 0 ; i < row; i ++)
  {
    for (int j = 0 ; j < col ; j ++)
    {
      if (sqrt(pow(Re.at<float>(i,j),2) + pow(Im.at<float>(i,j),2)) > max) 
      {
        max = sqrt(pow(Re.at<float>(i,j),2) + pow(Im.at<float>(i,j),2));
      }
    }
  }


   float T = epsilon * max;  

 // Creation of Q

  for (int i = 0 ; i < row; i ++)
  {
    for (int j = 0 ; j < col ; j ++)
    {
      if (sqrt(pow(Re.at<float>(i,j),2) + pow(Im.at<float>(i,j),2)) > T)
      {
        Re.at<float>(i,j) = Re.at<float>(i,j)/(pow(Re.at<float>(i,j),2) + pow(Im.at<float>(i,j),2));
        Im.at<float>(i,j) = -Im.at<float>(i,j)/(pow(Re.at<float>(i,j),2) + pow(Im.at<float>(i,j),2));
      }
      else 
      {
        Re.at<float>(i,j) = 1/T;
        Im.at<float>(i,j) = 0; 
      }
    }
  }


  vector<Mat> complexChannel_inverse;

  complexChannel_inverse.push_back(Re); 
  complexChannel_inverse.push_back(Im); 

  Mat Q = Mat::zeros(Re.rows, Re.cols, CV_32F); 

  merge(complexChannel_inverse,Q);


  // Multiplication of the inverse filter

  Mat restorated_ft;

  mulSpectrums(degraded_ft, Q, restorated_ft, 1); 

  // Creation of the restorated image

  Mat restorated; 


  dft(restorated_ft, restorated, CV_DXT_INV_SCALE);

  // Take only the real part of the restorated image - after the inverse fourier transform
  split(restorated, complexChannel);
  restorated = complexChannel[0];

  //Threshold the restorated image (values between 0 and 255)
  threshold(restorated, restorated, 255, 255, CV_THRESH_TRUNC);
  threshold(restorated, restorated, 0, 0, CV_THRESH_TOZERO);


   return restorated;
}

// Function applies wiener filter to restorate a degraded image
/*
degraded :  degraded input image
filter   :  filter which caused degradation
snr      :  signal to noise ratio of the input image
return   :   restorated output image
*/
Mat Dip4::wienerFilter(Mat& degraded, Mat& filter, double snr){
	
  // Creation of a shifted filter with degraded image size

  int d_row = degraded.rows; 
  int d_col = degraded.cols; 

  Mat filter_resize = Mat::zeros(d_row, d_col, CV_32FC1); 

  for (int i = 0 ; i < filter.rows ; i ++)
  {
    for (int j = 0 ; j < filter.cols ; j++)
    {
      filter_resize.at<float>(i,j) = filter.at<float>(i,j); 
    }
  }

  filter_resize = circShift(filter_resize, -filter.rows/2, -filter.cols/2);

  // Fourier transform of the degraded image and of the (resized and shifted) filter

  Mat degraded_ft; 
  Mat filter_ft; 

  dft(degraded, degraded_ft, DFT_COMPLEX_OUTPUT);
  dft(filter_resize, filter_ft,  DFT_COMPLEX_OUTPUT);

  // Separation of real and imaginary part of the filter in frequency domain

  Mat complexChannel[2];
  split(filter_ft, complexChannel);

  Mat Re = complexChannel[0];
  Mat Im = complexChannel[1]; 

  // calculation of the threshold

  float epsilon = 0.05; 

  int row = filter_ft.rows;
  int col = filter_ft.cols;

  float max = 0; 

  for (int i = 0 ; i < row; i ++)
  {
    for (int j = 0 ; j < col ; j++)
    {
      if (sqrt(pow(Re.at<float>(i,j),2) + pow(Im.at<float>(i,j),2)) > max) 
      {
        max = sqrt(pow(Re.at<float>(i,j),2) + pow(Im.at<float>(i,j),2));
      }
    }
  }

   float T = epsilon * max; 
 
 
 // Creation of Q

  for (int i = 0 ; i < row; i ++)
  {
    for (int j = 0 ; j < col ; j++)
    {
        Re.at<float>(i,j) = Re.at<float>(i,j)/(pow(Re.at<float>(i,j),2) + pow(Im.at<float>(i,j),2) + 1/pow(snr,2));
        Im.at<float>(i,j) = -Im.at<float>(i,j)/(pow(Re.at<float>(i,j),2) + pow(Im.at<float>(i,j),2) + 1/pow(snr,2));
    }
  }


  vector<Mat> complexChannel_inverse;

  complexChannel_inverse.push_back(Re); 
  complexChannel_inverse.push_back(Im); 

  Mat Q = Mat::zeros(Re.rows, Re.cols, CV_32F); 

  merge(complexChannel_inverse,Q);


  // Multiplication of the inverse filter

  Mat restorated_ft;

  mulSpectrums(degraded_ft, Q, restorated_ft, 1); 

  // Creation of the restorated image

  Mat restorated; 


  dft(restorated_ft, restorated, CV_DXT_INV_SCALE);

  // Take only the real part of the restorated image - after the inverse fourier transform
  split(restorated, complexChannel);
  restorated = complexChannel[0];

  //Threshold the restorated image (values between 0 and 255)
  threshold(restorated, restorated, 255, 255, CV_THRESH_TRUNC);
  threshold(restorated, restorated, 0, 0, CV_THRESH_TOZERO);



   return restorated;
}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

// function calls processing function
/*
in                   :  input image
restorationType     :  integer defining which restoration function is used
kernel               :  kernel used during restoration
snr                  :  signal-to-noise ratio (only used by wieder filter)
return               :  restorated image
*/
Mat Dip4::run(Mat& in, string restorationType, Mat& kernel, double snr){

   if (restorationType.compare("wiener")==0){
      return wienerFilter(in, kernel, snr);
   }else{
      return inverseFilter(in, kernel);
   }

}

// Function degrades a given image with gaussian blur and additive gaussian noise
/*
img         :  input image
degradedImg :  degraded output image
filterDev   :  standard deviation of kernel for gaussian blur
snr         :  signal to noise ratio for additive gaussian noise
return      :  the used gaussian kernel
*/
Mat Dip4::degradeImage(Mat& img, Mat& degradedImg, double filterDev, double snr){

    int kSize = round(filterDev*3)*2 - 1;
   
    Mat gaussKernel = getGaussianKernel(kSize, filterDev, CV_32FC1);
    gaussKernel = gaussKernel * gaussKernel.t();

    Mat imgs = img.clone();
    dft( imgs, imgs, CV_DXT_FORWARD, img.rows);
    Mat kernels = Mat::zeros( img.rows, img.cols, CV_32FC1);
    int dx, dy; dx = dy = (kSize-1)/2.;
    for(int i=0; i<kSize; i++) for(int j=0; j<kSize; j++) kernels.at<float>((i - dy + img.rows) % img.rows,(j - dx + img.cols) % img.cols) = gaussKernel.at<float>(i,j);
	dft( kernels, kernels, CV_DXT_FORWARD );
	mulSpectrums( imgs, kernels, imgs, 0 );
	dft( imgs, degradedImg, CV_DXT_INV_SCALE, img.rows );
	
    Mat mean, stddev;
    meanStdDev(img, mean, stddev);

    Mat noise = Mat::zeros(img.rows, img.cols, CV_32FC1);
    randn(noise, 0, stddev.at<double>(0)/snr);
    degradedImg = degradedImg + noise;
    threshold(degradedImg, degradedImg, 255, 255, CV_THRESH_TRUNC);
    threshold(degradedImg, degradedImg, 0, 0, CV_THRESH_TOZERO);

    return gaussKernel;
}

// Function displays image (after proper normalization)
/*
win   :  Window name
img   :  Image that shall be displayed
cut   :  whether to cut or scale values outside of [0,255] range
*/
void Dip4::showImage(const char* win, Mat img, bool cut){

   Mat tmp = img.clone();

   if (tmp.channels() == 1){
      if (cut){
         threshold(tmp, tmp, 255, 255, CV_THRESH_TRUNC);
         threshold(tmp, tmp, 0, 0, CV_THRESH_TOZERO);
      }else
         normalize(tmp, tmp, 0, 255, CV_MINMAX);
         
      tmp.convertTo(tmp, CV_8UC1);
   }else{
      tmp.convertTo(tmp, CV_8UC3);
   }
   imshow(win, tmp);
}

// function calls some basic testing routines to test individual functions for correctness
void Dip4::test(void){

   test_circShift();
   cout << "Press enter to continue"  << endl;
   cin.get();

}

void Dip4::test_circShift(void){

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
      cout << "ERROR: Dip4::circShift(): Result of circshift seems to be wrong!" << endl;
      return;
   }
   cout << "Message: Dip4::circShift() seems to be correct" << endl;
}

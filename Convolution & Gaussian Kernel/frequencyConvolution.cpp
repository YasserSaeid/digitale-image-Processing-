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
%%writefile Blur.cu
#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <sys/time.h>

using namespace cv;

Mat *get_sharpen_image(Mat *, Mat *, Mat *, Mat *, int , Size  );
Mat *get_blur_image(Mat *, Mat *, int , Size );
Mat *get_high_pass_image(Mat *, Mat *, Mat *, int , Size );

void matToUchar(Mat, uchar *, int, int);
void ucharToMat(uchar *, Mat, int, int);

void matToUchar(Mat frame, uchar *uFrame, int width, int height)
{
    for (int ch = 0; ch < 3; ch++)
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                uFrame[(ch * width * height) + (i * width + j)] = frame.at<Vec3b>(i, j)[ch];
               
}

// Function to cast uchar to Mat
void ucharToMat(uchar *uFrame, Mat frame, int width, int height)
{
    for (int ch = 0; ch < 3; ch++)
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                frame.at<Vec3b>(i, j)[ch] = uFrame[(ch * width * height) + (i * width + j)];
}


__global__ void get_blur_image(uchar *d_original_image, uchar *d_blur_image, int width, int height, int nThreads){
    
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    // Position variables to get the optical flow
    int startPos = (thread_id < (width * height) % nThreads) ? ((width * height) / nThreads) * thread_id + thread_id : ((width * height) / nThreads) * thread_id + (width * height) % nThreads;
    int endPos = (thread_id < (width * height) % nThreads) ? startPos + ((width * height) / nThreads) : startPos + ((width * height) / nThreads) - 1; 
    int i = (startPos / width), j = (startPos % width);

    int filterBlur[25] = {1,4,7,4,1,4,16,26,16,4,7,26,41,26,7,4,16,26,16,4,1,4,7,4,1};

    for (; startPos <= endPos; startPos++){
        if(i <= 1 || j <= 1 || i >= height - 1 || j >= width - 1 ){

           //Blue
           *(d_blur_image + (i * width + j)) = (int) *(d_original_image + (i * width + j)) / 3;
           //Green
           *(d_blur_image + (width * height) + (i * width + j)) = (int) *(d_original_image + (i * width + j)) / 3;
           //Red
           *(d_blur_image + (2 * width * height) + (i * width + j)) = (int) *(d_original_image + (i * width + j)) / 3;

        }else{
              int positionFilter = 0;
              int blueBlur = 0;
              int greenBlur = 0;
              int redBlur = 0;
              for(int k = -12; k <= 12 ; k++){
                  blueBlur += *(d_original_image + (i * width + j) + k) * filterBlur[positionFilter];
                  greenBlur += *(d_original_image + (width * height) + (i * width + j) + k) * filterBlur[positionFilter];
                  redBlur += *(d_original_image + (2 * width * height) + (i * width + j) + k) * filterBlur[positionFilter];
                  positionFilter++;
              }
              *(d_blur_image + (i * width + j)) = (int) blueBlur / 273;
              *(d_blur_image + (width * height) + (i * width + j)) = (int) greenBlur / 273;
              *(d_blur_image + (2 * width * height) + (i * width + j)) = (int) redBlur / 273;
        }
        j += 1;
        if (j == width){
            i += 1;
            j = 0;
        }
     }
}


int main(int argc, char** argv )
{

    // Declare the variables for time measurement
    struct timeval tval_before, tval_after, tval_result;

    Mat image;
    //Mat *result_sharpen;
    Mat imageChannel[3];    
    Mat saveImage;
    Size frameSize;
    
    cudaError_t err = cudaSuccess;
    
    std::vector<Mat> mChannels;

    if ( argc != 2 ){
        printf("usage: test <Image_Path>\n");
        return -1;
    }
    // Get start time
    gettimeofday(&tval_before, NULL);

    image = imread(argv[1], 1);
    frameSize = image.size();

    if ( !image.data ){
        printf("No image data \n");
        return -1;
    }

    split(image, imageChannel);

    int width = frameSize.width;
    int height = frameSize.height;
    int channels = 3;
    
    int size = height * width * channels * sizeof(uchar);


    uchar *d_original_image, *d_blur_image;
    int nThreads = 64;
    /*Variable para el número de bloques*/
    int nBlocks = 40;

    uchar *originalImage = (uchar *)malloc(size);
    uchar *blurImage = (uchar *)malloc(size);

    matToUchar(image, originalImage, width, height);
    
    //Inputs
    err = cudaMalloc((void **)&d_original_image, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device d_original_image (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_original_image, originalImage, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy ker from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Outputs
    err = cudaMalloc((void **)&d_blur_image, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device d_original_image (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    get_blur_image<<<nBlocks, nThreads>>>(d_original_image, d_blur_image, width, height, nBlocks * nThreads);
    cudaDeviceSynchronize();

    err = cudaMemcpy(blurImage, d_blur_image, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy blurImage from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    Mat imageBlur = Mat::zeros(Size(width, height), CV_8UC3);

    ucharToMat(blurImage, imageBlur, width, height);

    // Calcular los tiempos en tval_result
    //  Get end time
    gettimeofday(&tval_after, NULL);

    timersub(&tval_after, &tval_before, &tval_result);
    /*Imprimir informe*/
    printf("------------------------------------------------------------------------------\n");
    printf("Tiempo de ejecución: %ld.%06ld s \n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);

    //mChannels = {imageBlur[0],imageBlur[1],imageBlur[2]};
    //merge(mChannels, saveImage);

    if(imwrite("blur.png", imageBlur) == false){
        std::cout << "Saving fail"<<std::endl;
        return -1;
    }

    waitKey(0);

    return 0;
}
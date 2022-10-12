#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <sys/time.h>

using namespace cv;

Mat *get_sharpen_image(Mat *, Mat *, Mat *, Mat *, int , Size  );
Mat *get_blur_image(Mat *, Mat *, int , Size );
Mat *get_high_pass_image(Mat *, Mat *, Mat *, int , Size );

int main(int argc, char** argv )
{
    // Declare the variables for time measurement
    struct timeval tval_before, tval_after, tval_result;

    Mat image;
    Mat *result_sharpen;
    Mat imageChannel[3];
    Mat imageBlur[3];
    Mat imageHighPass[3];
    Mat imageSharpen[3];
    
    Mat saveImage;
    Size frameSize;
    
    std::vector<Mat> mChannels;

    if ( argc != 2 ){
        printf("usage: DisplayImage.out <Image_Path>\n");
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
    split(image, imageBlur);
    split(image, imageHighPass);
    split(image, imageSharpen);
    
    result_sharpen = get_sharpen_image(imageChannel, imageBlur, imageHighPass, imageSharpen, 4, frameSize);
    // Calcular los tiempos en tval_result
    //  Get end time
    gettimeofday(&tval_after, NULL);

    timersub(&tval_after, &tval_before, &tval_result);
    /*Imprimir informe*/
    printf("------------------------------------------------------------------------------\n");
    printf("Tiempo de ejecución: %ld.%06ld s \n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);

    mChannels = {result_sharpen[0],result_sharpen[1],result_sharpen[2]};

    merge(mChannels, saveImage);

    if(imwrite("sharpen.png", saveImage) == false){
        std::cout << "Saving fail"<<std::endl;
        return -1;
    }

    imshow("Sharpen image", saveImage);

    waitKey(0);

    return 0;
}

Mat *get_sharpen_image(Mat *original_image, Mat *blur_image, Mat *high_pass_image, Mat *sharpen_image, int num_threads, Size frameSize ){
    
    blur_image = get_blur_image(original_image, blur_image, num_threads, frameSize);
    high_pass_image = get_high_pass_image(original_image, blur_image, high_pass_image, num_threads, frameSize);
    
    int k = 2;
    for(int i = 0; i < frameSize.height; i++){
        for(int j = 0; j < frameSize.width; j++){
            sharpen_image[0].at<uchar>(i,j) = original_image[0].at<uchar>(i,j) + (k * high_pass_image[0].at<uchar>(i,j));
            sharpen_image[1].at<uchar>(i,j) = original_image[1].at<uchar>(i,j) + (k * high_pass_image[1].at<uchar>(i,j));
            sharpen_image[2].at<uchar>(i,j) = original_image[2].at<uchar>(i,j) + (k * high_pass_image[2].at<uchar>(i,j)); 
        }
    }

    return high_pass_image;
}

Mat *get_blur_image(Mat *imageChannel, Mat *imageBlur, int thread_count, Size frameSize){
    int filterBlur[25] = {1,4,7,4,1,4,16,26,16,4,7,26,41,26,7,4,16,26,16,4,1,4,7,4,1};
    #pragma omp parallel num_threads(thread_count)
    {
        for(int i = 0; i < frameSize.height; i++){
                #pragma omp for
                for(int j = 0; j < frameSize.width; j++){
                    if(i <= 1 || j <= 1 || i >= frameSize.height - 1 || j >= frameSize.width - 1 ){
                        //Blue
                        imageBlur[0].at<uchar>(i,j) = imageChannel[0].at<uchar>(i,j)/3;
                        //Green
                        imageBlur[1].at<uchar>(i,j) = imageChannel[1].at<uchar>(i,j)/3;
                        //Red
                        imageBlur[2].at<uchar>(i,j) = imageChannel[2].at<uchar>(i,j)/3;  
                    }else{
                        int positionFilter = 0;
                        int blueBlur = 0;
                        int greenBlur = 0;
                        int redBlur = 0;
                        for(int k = -2; k <= 2 ; k++){
                            for(int l = -2; l <= 2; l++){
                                blueBlur += imageChannel[0].at<uchar>(i+k,j+l) * filterBlur[positionFilter];
                                greenBlur += imageChannel[1].at<uchar>(i+k,j+l) * filterBlur[positionFilter];
                                redBlur += imageChannel[2].at<uchar>(i+k,j+l) * filterBlur[positionFilter];
                                positionFilter++;
                            }
                        }
                        imageBlur[0].at<uchar>(i,j) = (int) blueBlur / 273;
                        imageBlur[1].at<uchar>(i,j) = (int) greenBlur / 273;
                        imageBlur[2].at<uchar>(i,j) = (int) redBlur / 273;

                    }
                }
            }
    } 
    return imageBlur;
}

Mat *get_high_pass_image(Mat *imageChannel, Mat *imageBlur, Mat *imageHighPass, int num_threads, Size frameSize){
    for(int i = 0; i < frameSize.height; i++){
        for(int j = 0; j < frameSize.width; j++){
            imageHighPass[0].at<uchar>(i,j) = imageChannel[0].at<uchar>(i,j) - imageBlur[0].at<uchar>(i,j);
            imageHighPass[1].at<uchar>(i,j) = imageChannel[1].at<uchar>(i,j) - imageBlur[1].at<uchar>(i,j);
            imageHighPass[2].at<uchar>(i,j) = imageChannel[2].at<uchar>(i,j) - imageBlur[2].at<uchar>(i,j); 
        }
    }
    return imageHighPass;
}

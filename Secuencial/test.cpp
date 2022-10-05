#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
int main(int argc, char** argv )
{
    Mat image;
    Mat imageChannel[3];
    Mat imageBlur[3];
    Mat imageHighPass[3];
    Mat imageSharpen[3];
    
    Mat saveImage;
    Size frameSize;

    int filterBlur[9] = {1,2,1,2,4,2,1,2,1}; 
    
    std::vector<Mat> mChannels;

    if ( argc != 2 ){
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

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
    

    for(int i = 0; i < frameSize.height; i++){
        for(int j = 0; j < frameSize.width; j++){
            if(i < 1 || j < 1 || i > frameSize.height - 1 || j > frameSize.width - 1 ){
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
                for(int k = -1; k <= 1 ; k++){
                    for(int l = -1; l <= 1; l++){
                        blueBlur += imageChannel[0].at<uchar>(i+k,j+l) * filterBlur[positionFilter];
                        greenBlur += imageChannel[1].at<uchar>(i+k,j+l) * filterBlur[positionFilter];
                        redBlur += imageChannel[2].at<uchar>(i+k,j+l) * filterBlur[positionFilter];
                        positionFilter++;
                    }
                }
                imageBlur[0].at<uchar>(i,j) = (int) blueBlur / 16;
                imageBlur[1].at<uchar>(i,j) = (int) greenBlur / 16;
                imageBlur[2].at<uchar>(i,j) = (int) redBlur / 16;

            }
        }
    }

    for(int i = 0; i < frameSize.height; i++){
        for(int j = 0; j < frameSize.width; j++){
            imageHighPass[0].at<uchar>(i,j) = imageChannel[0].at<uchar>(i,j) - imageBlur[0].at<uchar>(i,j);
            imageHighPass[1].at<uchar>(i,j) = imageChannel[1].at<uchar>(i,j) - imageBlur[1].at<uchar>(i,j);
            imageHighPass[2].at<uchar>(i,j) = imageChannel[2].at<uchar>(i,j) - imageBlur[2].at<uchar>(i,j); 
        }
    }
    int k = 2;
    for(int i = 0; i < frameSize.height; i++){
        for(int j = 0; j < frameSize.width; j++){
            imageSharpen[0].at<uchar>(i,j) = imageChannel[0].at<uchar>(i,j) + (k * imageHighPass[0].at<uchar>(i,j));
            imageSharpen[1].at<uchar>(i,j) = imageChannel[1].at<uchar>(i,j) + (k * imageHighPass[1].at<uchar>(i,j));
            imageSharpen[2].at<uchar>(i,j) = imageChannel[2].at<uchar>(i,j) + (k * imageHighPass[2].at<uchar>(i,j)); 
        }
    }

    mChannels = {imageSharpen[0],imageSharpen[1],imageSharpen[2]};

    merge(mChannels, saveImage);

    if(imwrite("sharpen.png", saveImage) == false){
        std::cout << "Saving fail"<<std::endl;
        return -1;
    }

    imshow("Sharpen image", saveImage);

    waitKey(0);

    return 0;
}
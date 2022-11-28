#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <sys/time.h>
#include <mpi.h>

using namespace cv;

void matToUchar(Mat frame, uchar *uFrame, int width, int height)
{
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            for (int ch = 0; ch < 3; ch++)
                uFrame[(3 * i * width) + (3 * j) + ch] = frame.at<Vec3b>(i, j)[ch];
               
}

void ucharToMat(uchar *uFrame, Mat frame, int width, int height)
{
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            for (int ch = 0; ch < 3; ch++)
                frame.at<Vec3b>(i, j)[ch] = uFrame[(3 * i * width) + (3 * j) + ch];
}

float lerp(float a, float b, float f) 
{
    return (a * (1.0 - f)) + (b * f);
}

/*void get_sharpen_image(Mat *original_image, Mat *blur_image, Mat *sharpen_image, Size frameSize){      
    int k = 2;
    for(int i = 0; i < frameSize.height; i++){
        for(int j = 0; j < frameSize.width; j++){
            float blueSharpen = lerp((float) blur_image[0].at<uchar>(i,j), (float) original_image[0].at<uchar>(i,j), (float) 1 + k);
            float greenSharpen = lerp((float) blur_image[1].at<uchar>(i,j), (float) original_image[1].at<uchar>(i,j), (float) 1 + k);
            float redSharpen = lerp((float) blur_image[2].at<uchar>(i,j), (float) original_image[2].at<uchar>(i,j), (float) 1 + k);
            sharpen_image[0].at<uchar>(i,j) = (int) blueSharpen;
            sharpen_image[1].at<uchar>(i,j) = (int) greenSharpen;
            sharpen_image[2].at<uchar>(i,j) = (int) redSharpen; 
        }
    }
}*/

void get_sharpen_image(uchar *original_image, uchar *blur_image, uchar *sharpen_image, Size frameSize){      
    int k = 2;
    int height = frameSize.height;
    int width = frameSize.width;
    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            float blueSharpen = lerp((float) *(blur_image + (3 * i * width) + (3 * j)), (float) *(original_image + (3 * i * width) + (3 * j)), (float) 1 + k);
            float greenSharpen = lerp((float) *(blur_image + (3 * i * width) + (3 * j) + 1), (float) *(original_image + (3 * i * width) + (3 * j) + 1), (float) 1 + k);
            float redSharpen = lerp((float) *(blur_image + (3 * i * width) + (3 * j) + 2), (float) *(original_image + (3 * i * width) + (3 * j) + 2), (float) 1 + k);  
            *(sharpen_image + (3 * i * width) + (3 * j)) = (int) blueSharpen;
            *(sharpen_image + (3 * i * width) + (3 * j) + 1) = (int) greenSharpen;
            *(sharpen_image + (3 * i * width) + (3 * j) + 2) = (int) redSharpen;
        } 
    }               
}

void get_blur_image(uchar *imageChannel, uchar *imageBlur, Size frameSize, int startPos, int endPos, int processId){
    int filterBlur[25] = {1,4,7,4,1,4,16,26,16,4,7,26,41,26,7,4,16,26,16,4,1,4,7,4,1};
    int height = frameSize.height;
    int width = frameSize.width;
    int iterSize = endPos - startPos + 1;
    int i = (startPos / width), j = (startPos % width);

    /*std::cout<<"ProcessId: "<<processId<<std::endl;
    std::cout<<"startPos: "<<startPos<<std::endl;
    std::cout<<"endPos: "<<endPos<<std::endl;
    std::cout<<"width: "<<width<<std::endl;
    std::cout<<"height: "<<height<<std::endl;
    std::cout<<"iterSize: "<<iterSize * sizeof(uchar)<<std::endl;*/

    for (int iter = 0; startPos <= endPos; startPos++){
        if(i <= 1 || j <= 1 || i >= height - 1 || j >= width - 1 ){
            //Blue
            *(imageBlur + iter) = (int) *(imageChannel + (3 * i * width) + (3 * j)) / 3;
            //Green
            *(imageBlur + iter + 1) = (int) *(imageChannel + (3 * i * width) + (3 * j) + 1) / 3;
            //Red
            *(imageBlur + iter + 2) = (int) *(imageChannel + (3 * i * width) + (3 * j) + 2) / 3;
        }else{
            int positionFilter = 0;
            int blueBlur = 0;
            int greenBlur = 0;
            int redBlur = 0;
            for(int k = -2; k <= 2 ; k++){
                for(int l = -2; l <= 2; l++){
                    blueBlur += *(imageChannel + (3 * (i + k) * width) + (3 * (j + l))) * filterBlur[positionFilter];
                    greenBlur += *(imageChannel + (3 * (i + k) * width) + (3 * (j + l)) + 1) * filterBlur[positionFilter];
                    redBlur += *(imageChannel + (3 * (i + k) * width) + (3 * (j + l)) + 2 ) * filterBlur[positionFilter];
                    positionFilter++;
                }
            }
            *(imageBlur + iter) = (int) (blueBlur / 273);
            *(imageBlur + iter + 1) = (int) (greenBlur / 273);
            *(imageBlur + iter + 2) = (int) (redBlur / 273);
        }
        j += 1;
        iter += 3;
        if (j == width){
            i += 1;
            j = 0;
        }
    }   
}

int main(int argc, char** argv )
{
    int processId, nProcess;

    double totalTime = 0.0, clusterTime = 0.0, processTime = 0.0, timeStart = 0.0, timeEnd = 0.0;

    VideoCapture loadVideo;
    Mat image;
    Mat *result_sharpen;
    Mat imageChannel[3];
    Mat imageBlur[3];
    Mat imageHighPass[3];
    Mat imageSharpen[3]; 
    Mat saveFrame, newFrame; 
    std::vector<Mat> mChannels;
    int thread_num;
    int width, height, fps;
    VideoWriter saveVideo;
    int size;
    int frame = 0, frameCount = 0;

    uchar *originalImage;


    if ( argc != 2 ){
        printf("usage: DisplayImage.out <Video_Path> \n");
        return -1;
    }
    
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcess);
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);

    if (processId == 0){
        loadVideo = VideoCapture(argv[1]);
        if (!loadVideo.isOpened()){
            std::cout << "Error opening video stream or file" << std::endl;
            return -1;
        }
        frameCount = loadVideo.get(CAP_PROP_FRAME_COUNT);
        fps = loadVideo.get(CAP_PROP_FPS);
        width = loadVideo.get(CAP_PROP_FRAME_WIDTH), height = loadVideo.get(CAP_PROP_FRAME_HEIGHT);
        saveVideo = VideoWriter("myVideo2.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(width, height));
        size = height * width * 3 * sizeof(uchar);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(&fps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&frameCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
 
    while(frame < frameCount){
        processTime = 0.0;
        clusterTime = 0.0;

        originalImage = (uchar *)malloc(size);
        if (processId == 0){    
            loadVideo.read(newFrame);                       
            matToUchar(newFrame, originalImage, width, height);              
        }

        MPI_Bcast(&originalImage, 1, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

        int startPos = (processId < (width * height) % nProcess) ? ((width * height) / nProcess) * processId + processId : ((width * height) / nProcess) * processId + (width * height) % nProcess;
        int endPos = (processId < (width * height) % nProcess) ? startPos + ((width * height) / nProcess) : startPos + ((width * height) / nProcess) - 1; 
        int sizeArrays = ((endPos - startPos) + 1);
        uchar *blurImageResp = (uchar *)malloc(size);
        uchar *blurImage = (uchar *)malloc(sizeArrays * 3 *sizeof(uchar));       
        
        timeStart = MPI_Wtime();
        get_blur_image(originalImage, blurImage, Size(width, height), startPos, endPos, processId);
        timeEnd = MPI_Wtime();
        processTime += fabs(timeEnd - timeStart);

        MPI_Barrier(MPI_COMM_WORLD);  

        MPI_Gather(blurImage, sizeArrays * 3 *sizeof(uchar), MPI_UNSIGNED_CHAR, blurImageResp, sizeArrays * 3 *sizeof(uchar), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        
        MPI_Reduce(&processTime, &clusterTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if(processId == 0){
            timeStart = MPI_Wtime();

            uchar *sharpenResp = (uchar *)malloc(size);
            Mat imageSharpenMat = Mat::zeros(Size(width, height), CV_8UC3);
            get_sharpen_image(originalImage, blurImageResp, sharpenResp, Size(width, height));
            
            timeEnd = MPI_Wtime();

            clusterTime += fabs(timeEnd - timeStart);
            clusterTime = clusterTime / nProcess;

            totalTime += clusterTime;

            ucharToMat(sharpenResp, imageSharpenMat, width, height);
            saveVideo.write(imageSharpenMat);
        }
        frame++;
    }
    saveVideo.release();

    if (processId == 0){
        std::cout<<"Total time: "<<totalTime<<std::endl;
    }

    MPI_Finalize();    
    return 0;
}



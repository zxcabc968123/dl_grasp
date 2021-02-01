#include<ros/ros.h>
#include<iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include<sensor_msgs/Image.h>

//#include<std_msg/img.h>
using namespace cv;
/*
void callback(const sensor_msgs::ImageConstPtr& msg)//當收到資訊時執行這個副函式
{
    ros::NodeHandle nh;
    ros::Publisher pub=nh.advertise<sensor_msgs::Image>("/zafterimage",100);
    //ROS_INFO("123");
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
    
    cv::Mat tmp;
    cv::Mat img;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    dilate(cv_ptr->image, tmp, element,Point(-1,-1),4);
    erode(tmp, tmp, element,Point(-1,-1),2);
    cv::convertScaleAbs(tmp,img,0.5666);
    int k;
    for( int i=0 ; i<640 ; i++ )
    {
        for(int j=0 ; j<480 ; j++)
        {
            if( img.at<unsigned char>(j,i) <= 80 )
            {
                //srand( time(NULL) );
                k=(i+j)%3+243;
                //printf("The Random Number is %d .\n", k);
                img.at<unsigned char>(j,i)=k;
            }
        }
    }
    dilate(img, img, element,Point(-1,-1),1);
    erode(img, cv_ptr->image, element,Point(-1,-1),1);
    //cv::imshow("OPENCV_WINDOW", cv_ptr->image);
    //cv::waitKey(1);
    /////////////////////////////////////////
    pub.publish(cv_ptr->toImageMsg());
}
*/
class ImageConverter
{
            
    public:
        
        cv::Mat tmp;
        cv::Mat img;
        cv::Mat preimage;

        cv_bridge::CvImagePtr cv_ptr;
        int k;
        double psnr;
        void callback(const sensor_msgs::ImageConstPtr& msg);
        double getpsnr(const Mat& I2);
};
    double ImageConverter::getpsnr(const Mat& I2)
    {
        Mat s1;
        const char* imageName = "/home/allen/dl_grasp/src/data_expend/background/background_7.jpg";
        Mat image = imread( imageName, 0 );
        absdiff(image, I2, s1);       // |I1 - I2|AbsDiff函数是 OpenCV 中计算两个数组差的绝对值的函数
        s1.convertTo(s1, CV_32F);  // 这里我们使用的CV_32F来计算，因为8位无符号char是不能进行平方计算
        s1 = s1.mul(s1);           // |I1 - I2|^2

        Scalar s = sum(s1);         //对每一个通道进行加和

        //double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels
        double sse = s.val[0];
        if( sse <= 1e-10) // 对于非常小的值我们将约等于0
            return 0;
        else
        {
            double  mse =sse /(double)(image.channels() * image.total());//计算MSE
            double psnr = 10.0 * log10((245.0 * 245.0) / mse);
            //ROS_INFO("psnr : %f",psnr);
            return psnr;//返回PSNR
        }
    };
    void ImageConverter::callback(const sensor_msgs::ImageConstPtr& msg)
    {
        
        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        ros::Rate loop_rate(100);
        
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
        dilate(cv_ptr->image, img, element,Point(-1,-1),2);
        erode(img, img, element,Point(-1,-1),2);
        cv::convertScaleAbs(img,img,0.57);
        ////
        
        ////
        int k;
        for( int i=0 ; i<640 ; i++ )
        {
            for(int j=0 ; j<480 ; j++)
            {
                if( img.at<unsigned char>(j,i) <= 80 )
                {
                    k=(i+j)%3+243;
                    img.at<unsigned char>(j,i)=k;
                }
            }
            //printf("%d   ",img.at<unsigned char>(10,10));
        }
        dilate(img, img, element,Point(-1,-1),1);
        erode(img, this->preimage, element,Point(-1,-1),1);
        //this->psnr = this->getpsnr(this->preimage);
        // pub.publish(cv_ptr->toImageMsg());
        // loop_rate.sleep();
        //ROS_INFO("psnr : %f",this->psnr);
        // cv::imshow("OPENCV_WINDOW", this->preimage);
        // cv::waitKey(1);
        this->preimage=img;
    };
int main( int argc,char** argv)
{
    
    ros::init (argc,argv, "img_preprocess");//subscriber_node:node name
    ros::NodeHandle nh;
    ImageConverter a;
    ros::Publisher pub=nh.advertise<sensor_msgs::Image>("/zafterimage",10);
    ros::Subscriber sub=nh.subscribe("/camera/aligned_depth_to_color/image_raw",10,&ImageConverter::callback,&a);
    ros::Rate loop_rate(1000);

    sensor_msgs::ImagePtr num;
    std_msgs::Header header;
    while(ros::ok())
    {
        ros::spinOnce();
        /////////////////////////////////
        //p = a.preimage.at<unsigned char>(10,10);
        //ROS_INFO(a.preimage.at<unsigned char>(10,10));
        //printf("%d   ",a.preimage.at<unsigned char>(10,10));
        num = cv_bridge::CvImage(header, "mono8", a.preimage).toImageMsg();
        
        loop_rate.sleep();
        pub.publish(num);
        /////////////////////////////////
    }

    //ros::spin();
    //return 0;
}
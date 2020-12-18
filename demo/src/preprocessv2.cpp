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
        void callback(const sensor_msgs::ImageConstPtr& msg);
};
    void ImageConverter::callback(const sensor_msgs::ImageConstPtr& msg)
    {
        
        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        ros::Rate loop_rate(100);
        
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
        dilate(cv_ptr->image, img, element,Point(-1,-1),4);
        erode(img, img, element,Point(-1,-1),2);
        cv::convertScaleAbs(img,img,0.57);
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
        // pub.publish(cv_ptr->toImageMsg());
        // loop_rate.sleep();
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
    ros::Rate loop_rate(100);

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
        pub.publish(num);
        loop_rate.sleep();
        
        /////////////////////////////////
    }

    //ros::spin();
    //return 0;
}
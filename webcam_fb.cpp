#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <linux/fb.h>

#include <string>
#include <chrono>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <termios.h>
#include <unistd.h>

#include <dlib/dnn.h>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <sys/ioctl.h>


using namespace dlib;
using namespace std;


// ----------------------------------------------------------------------------------------
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

// ----------------------------------------------------------------------------------------

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;


// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image
                            >>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

struct framebuffer_info
{
    uint32_t bits_per_pixel;    // depth of framebuffer
    uint32_t xres_virtual;      // how many pixel in a row in virtual screen
};

struct framebuffer_info get_framebuffer_info ( const char *framebuffer_device_path );

std::vector<matrix<float,0,1>> features;

void add_features(string filename)
{
	ifstream in(filename, std::ios_base::binary);
	if(!in.good())
	{
		cout << filename << " error." << endl;
		in.close();
		return;
	}
	
	float f;
	
	for (int i = 0; i < 6; ++i)
	{
		dlib::array<float> float_array = dlib::array<float>();
		
		for (int j = 0; j < 128; ++j) {
			in >> f;
			float_array.push_back(f);
		}
		
		auto float_mat = mat(float_array);
		auto float_matrix = matrix<float, 0, 1>(float_mat);
		
		features.push_back(float_matrix);
	}
	
	in.close();
}

void init(){
    termios old;
    if (tcgetattr(0, &old) < 0)
        perror("tcsetattr()");
    
    old.c_lflag &= ~ICANON;
    old.c_lflag &= ~ECHO;
    old.c_cc[VMIN] = 0;
    old.c_cc[VTIME] = 0;    // 1/10 second
    if (tcsetattr(0, TCSANOW, &old) < 0)
            perror("tcsetattr ICANON");
}
void end(){
    termios old;
    if (tcgetattr(0, &old) < 0)
        perror("tcsetattr()");
    old.c_lflag |= ICANON;
    old.c_lflag |= ECHO;
    if (tcsetattr(0, TCSADRAIN, &old) < 0)
        perror ("tcsetattr ~ICANON");
}

int main()
{
    bool flag = 0;
    init();
	
    try
    {
		framebuffer_info fb_info = get_framebuffer_info("/dev/fb0");
		std::ofstream ofs("/dev/fb0");
		
        cv::VideoCapture cap(2);
        if (!cap.isOpened())
        {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }
		cap.set(CV_CAP_PROP_BUFFERSIZE, 1);
		
		int width = fb_info.xres_virtual * fb_info.bits_per_pixel / 8;
		int height = width * 0.6;
		cap.set(cv::CAP_PROP_FPS , 30);


        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("./shape_predictor_5_face_landmarks.dat") >> pose_model;

		anet_type net;
		deserialize("./metric_network_renset.dat") >> net;
		
		std::vector<cv::Point> points;
		std::vector<matrix<rgb_pixel>> images;
		matrix<rgb_pixel> image; 
		
		
		// add features
		add_features("features.bin");
		
		
		// Grab a frame
		cv::Mat frame;

		int i = 0;
		char c;
		
        // Grab and process frames until the main window is closed by the user.
        while(cap.read(frame))
        {
			// check type in c or not
			c = '\0';
			read(0, &c, 1);
			if(c == 'q'){
				cap.release();
				end();
				exit(0);
			}
            if(c == 'c'){
                flag = !flag;
            }
			
			if(flag)
			{
				images.clear();
				points.clear();
				cv_image<bgr_pixel> cimg(frame);
				
				// Detect faces  and find the pose of each face.
				for (auto face : detector(cimg))
				{
					// draw face edge
					cv::rectangle(frame, cv::Point(face.left(), face.top()), cv::Point(face.right(), face.bottom()), cv::Scalar(0, 0, 255), 1, 1, 0); 
					
					// get 5 landmarks
					full_object_detection landmarks = pose_model(cimg, face);
				
					// face clip
					extract_image_chip(cimg, get_face_chip_details(landmarks,150,0.25), image);
					
					images.push_back(move(image));
					points.push_back(cv::Point(face.left(), face.top() - 10));
					
					// draw left eyes
					//    38
					// 36    39
					//    41
					// 2 3
					//cv::rectangle(frame, cv::Point(landmarks.part(36).x(), landmarks.part(38).y()), cv::Point(landmarks.part(39).x(), landmarks.part(41).y()), cv::Scalar(0, 0, 255), 1, 1, 0);
					int temp = (landmarks.part(3).x() - landmarks.part(2).x()) / 2;
					cv::rectangle(frame, cv::Point(landmarks.part(2).x(), landmarks.part(2).y() - temp), cv::Point(landmarks.part(3).x(), landmarks.part(3).y() + temp), cv::Scalar(0, 0, 255), 1, 1, 0);
					
					// draw right eyes
					//    43
					// 42    45
					//    46
					// 1 0
					//cv::rectangle(frame, cv::Point(landmarks.part(42).x(), landmarks.part(43).y()), cv::Point(landmarks.part(45).x(), landmarks.part(46).y()), cv::Scalar(0, 0, 255), 1, 1, 0);
					temp = (landmarks.part(0).x() - landmarks.part(1).x()) / 2;
					cv::rectangle(frame, cv::Point(landmarks.part(1).x(), landmarks.part(1).y() - temp), cv::Point(landmarks.part(0).x(), landmarks.part(0).y() + temp), cv::Scalar(0, 0, 255), 1, 1, 0);
				}
			
				if (images.size() != 0)
				{
					auto t_start = std::chrono::high_resolution_clock::now(); 
					
					std::vector<matrix<float,0,1>> embedded = net(images);
					
					auto t_end = std::chrono::high_resolution_clock::now();
					cout << std::chrono::duration<double, std::milli>(t_end-t_start).count() / embedded.size() << " ms" << endl;
					
					for  (size_t i = 0; i < embedded.size(); ++i)
					{
						string label;
						if (length(embedded[i]-features[0]) < 0.5)
							label = "chiungi";
						else if (length(embedded[i]-features[1]) < 0.5)
							label = "chiungi";
						else if (length(embedded[i]-features[2]) < 0.5)
							label = "chiungi";
						else if (length(embedded[i]-features[3]) < 0.5)
							label = "tzuting";
						else if (length(embedded[i]-features[4]) < 0.5)
							label = "tzuting";
						else if (length(embedded[i]-features[5]) < 0.5)
							label = "tzuting";
						else
							label = "unknown" ;
						
						cv::putText(frame, label, points[i], cv::FONT_HERSHEY_COMPLEX, 2, cv::Scalar(0, 0, 255), 2, 8, 0);
					}
				}
			}
			
			cv::Size2f frame_size = frame.size();
			cv::cvtColor(frame, frame, 12);
			for ( int y = 0; y < frame_size.height; y++ )
			{
				// move to the next written position of output device framebuffer by "std::ostream::seekp()"
				// http://www.cplusplus.com/reference/ostream/ostream/seekp/
				ofs.seekp(y * fb_info.xres_virtual * fb_info.bits_per_pixel / 8 + (width - frame_size.width * fb_info.bits_per_pixel / 8) / 2);

				// write to the framebuffer by "std::ostream::write()"
				// you could use "cv::Mat::ptr()" to get the pointer of the corresponding row.
				// you also need to cacluate how many bytes required to write to the buffer
				// http://www.cplusplus.com/reference/ostream/ostream/write/
				// https://docs.opencv.org/3.4.7/d3/d63/classcv_1_1Mat.html#a13acd320291229615ef15f96ff1ff738
				ofs.write(reinterpret_cast<char *>(frame.ptr(y)), frame_size.width * fb_info.bits_per_pixel / 8);
			}
        }
    }
    catch(serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
        end();
    }
}


struct framebuffer_info get_framebuffer_info ( const char *framebuffer_device_path )
{
    struct framebuffer_info fb_info;        // Used to return the required attrs.
    struct fb_var_screeninfo screen_info;   // Used to get attributes of the device from OS kernel.

    // open deive with linux system call "open( )"
    // https://man7.org/linux/man-pages/man2/open.2.html
	int fd = open(framebuffer_device_path, O_RDWR);

    // get attributes of the framebuffer device thorugh linux system call "ioctl()"
    // the command you would need is "FBIOGET_VSCREENINFO"
    // https://man7.org/linux/man-pages/man2/ioctl.2.html
    // https://www.kernel.org/doc/Documentation/fb/api.txt
	ioctl(fd, FBIOGET_VSCREENINFO, &screen_info);

    // put the required attributes in variable "fb_info" you found with "ioctl() and return it."
    // fb_info.xres_virtual = ......
    // fb_info.bits_per_pixel = ......
	fb_info.xres_virtual = screen_info.xres_virtual;
    fb_info.bits_per_pixel = screen_info.bits_per_pixel;

    return fb_info;
};


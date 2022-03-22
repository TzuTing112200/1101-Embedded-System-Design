// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.  
    

    This example is essentially just a version of the face_landmark_detection_ex.cpp
    example modified to use OpenCV's VideoCapture object to read from a camera instead 
    of files.


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.  
*/
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <linux/fb.h>

#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <termios.h>
#include <unistd.h>

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

struct framebuffer_info
{
    uint32_t bits_per_pixel;    // depth of framebuffer
    uint32_t xres_virtual;      // how many pixel in a row in virtual screen
};

struct framebuffer_info get_framebuffer_info ( const char *framebuffer_device_path );

int main()
{
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
		
		int width = fb_info.xres_virtual * fb_info.bits_per_pixel / 8;
		int height = width * 0.6;
		cap.set(cv::CAP_PROP_FPS , 30);

        //image_window win;

        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("/home/es/workspace/Source/shape_predictor_68_face_landmarks.dat") >> pose_model;
		
		// Grab a frame
		cv::Mat frame;

        // Grab and process frames until the main window is closed by the user.
        while(cap.read(frame))
        {
			// Turn OpenCV's Mat into something dlib can deal with.  Note that this just
			// wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
			// long as temp is valid.  Also don't do anything to temp that would cause it
			// to reallocate the memory which stores the image as that will make cimg
			// contain dangling pointers.  This basically means you shouldn't modify temp
			// while using cimg.
			cv_image<bgr_pixel> cimg(frame);

			// Detect faces 
			std::vector<rectangle> faces = detector(cimg);
			cout << "num " <<  faces.size() << endl;
			
			// Detect faces  and find the pose of each face.
			for (auto face : faces)
			{
				// draw face edge
				cv::rectangle(frame, cv::Point(face.left(), face.top()), cv::Point(face.right(), face.bottom()), cv::Scalar(0, 0, 255), 1, 1, 0); 
				
				//get 68 landmarks
				full_object_detection landmarks = pose_model(cimg, face);
				
				// draw left eyes
				//    38
				// 36    39
				//    41
				cv::rectangle(frame, cv::Point(landmarks.part(36).x(), landmarks.part(38).y()), cv::Point(landmarks.part(39).x(), landmarks.part(41).y()), cv::Scalar(0, 0, 255), 1, 1, 0);
				
				// draw right eyes
				//    43
				// 42    45
				//    46
				cv::rectangle(frame, cv::Point(landmarks.part(42).x(), landmarks.part(43).y()), cv::Point(landmarks.part(45).x(), landmarks.part(46).y()), cv::Scalar(0, 0, 255), 1, 1, 0);
			}

			// Display it all on the screen
			//win.clear_overlay();
			//win.set_image(cimg);
			//win.add_overlay(render_face_detections(shapes));
			
			//cv::imshow("final", frame);
			//cv::waitKey(0);
			
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


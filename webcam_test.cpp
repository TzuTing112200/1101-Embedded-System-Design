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

#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

using namespace dlib;
using namespace std;

int main()
{
    try
    {
        cv::VideoCapture cap(0);
        if (!cap.isOpened())
        {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }

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
			
			// Find the pose of each face.
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
			
			cv::imshow("final", frame);
			cv::waitKey('q');
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


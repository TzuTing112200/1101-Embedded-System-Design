#include <dlib/dnn.h>
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

#include <fstream>
#include <chrono>

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
//template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
//template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;


// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            //alevel2<
                            //alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image
                            /*>>*/>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

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
		//cap.set(CV_CAP_PROP_BUFFERSIZE, 1);

        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("/home/es/workspace/Source/shape_predictor_5_face_landmarks.dat") >> pose_model;

		anet_type net;
		deserialize("metric_network_renset_small.dat") >> net;
		
		std::vector<cv::Point> points;
		std::vector<matrix<rgb_pixel>> images;
		matrix<rgb_pixel> image; 
		
		
		// add features
		add_features("features_small.bin");
		
		// Grab a frame
		cv::Mat frame;

        // Grab and process frames until the main window is closed by the user.
        while(cap.read(frame))
        {
			images.clear();
			points.clear();
			
			// Turn OpenCV's Mat into something dlib can deal with.  Note that this just
			// wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
			// long as temp is valid.  Also don't do anything to temp that would cause it
			// to reallocate the memory which stores the image as that will make cimg
			// contain dangling pointers.  This basically means you shouldn't modify temp
			// while using cimg.
			cv_image<bgr_pixel> cimg(frame);
			
			// Find the pose of each face.
			for (auto face : detector(cimg))
			{
				//get 68 landmarks
				auto landmarks = pose_model(cimg, face);
				
				extract_image_chip(cimg, get_face_chip_details(landmarks,32,0.25), image);
				
				images.push_back(move(image));
				points.push_back(cv::Point(face.left(), face.top() - 10));
				
				
				// draw face edge
				cv::rectangle(frame, cv::Point(face.left(), face.top()), cv::Point(face.right(), face.bottom()), cv::Scalar(0, 255, 0), 3, 1, 0); 
				
				// draw left eyes
				//    38
				// 36    39
				//    41
				// 2 3
				//cv::rectangle(frame, cv::Point(landmarks.part(36).x(), landmarks.part(38).y()), cv::Point(landmarks.part(39).x(), landmarks.part(41).y()), cv::Scalar(0, 0, 255), 1, 1, 0);
				int temp = (landmarks.part(3).x() - landmarks.part(2).x()) / 2;
				cv::rectangle(frame, cv::Point(landmarks.part(2).x(), landmarks.part(2).y() - temp), cv::Point(landmarks.part(3).x(), landmarks.part(3).y() + temp), cv::Scalar(0, 0, 255), 2, 1, 0);
				
				// draw right eyes
				//    43
				// 42    45
				//    46
				// 1 0
				//cv::rectangle(frame, cv::Point(landmarks.part(42).x(), landmarks.part(43).y()), cv::Point(landmarks.part(45).x(), landmarks.part(46).y()), cv::Scalar(0, 0, 255), 1, 1, 0);
				temp = (landmarks.part(0).x() - landmarks.part(1).x()) / 2;
				cv::rectangle(frame, cv::Point(landmarks.part(1).x(), landmarks.part(1).y() - temp), cv::Point(landmarks.part(0).x(), landmarks.part(0).y() + temp), cv::Scalar(0, 0, 255), 2, 1, 0);
			}
			
			auto t_start = std::chrono::high_resolution_clock::now(); 
			
			std::vector<matrix<float,0,1>> embedded = net(images);
			
			auto t_end = chrono::high_resolution_clock::now();
			
			for  (size_t i = 0; i < embedded.size(); ++i)
			{
				string label;
				if (length(embedded[i]-features[0]) < 0.45)
					label = "chiungi";
				else if (length(embedded[i]-features[1]) < 0.45)
					label = "chiungi";
				else if (length(embedded[i]-features[2]) < 0.45)
					label = "chiungi";
				else if (length(embedded[i]-features[3]) < 0.45)
					label = "tzuting";
				else if (length(embedded[i]-features[4]) < 0.45)
					label = "tzuting";
				else if (length(embedded[i]-features[5]) < 0.45)
					label = "tzuting";
				else
					label = "unknown" ;
				
				cv::putText(frame, label, points[i], cv::FONT_HERSHEY_COMPLEX, 2, cv::Scalar(0, 255, 0), 2, 8, 0);
				cout << chrono::duration<double, milli>(t_end-t_start).count() / embedded.size() << " ms\t" << label << endl;
			}

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


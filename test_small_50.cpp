#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/misc_api.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <fstream>

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

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        cout << "./test.out /test.jpg" << endl;
        return 1;
    }    
	
    frontal_face_detector detector = get_frontal_face_detector();
	
	shape_predictor sp;
    deserialize("/home/es/workspace/Source/shape_predictor_68_face_landmarks.dat") >> sp;

    anet_type net;
    deserialize("metric_network_renset_50.dat") >> net;

    std::vector<matrix<rgb_pixel>> images;
	matrix<rgb_pixel> image; 
	
	
	// get features and label
	load_image(image, "chiungi1_s50.jpg");
	images.push_back(std::move(image));
	load_image(image, "chiungi2_s50.jpg");
	images.push_back(std::move(image));
	load_image(image, "chiungi3_s50.jpg");
	images.push_back(std::move(image));
	load_image(image, "tzuting1_s50.jpg");
	images.push_back(std::move(image));
	load_image(image, "tzuting2_s50.jpg");
	images.push_back(std::move(image));
	load_image(image, "tzuting3_s50.jpg");
	images.push_back(std::move(image));
	
	std::vector<matrix<float,0,1>> features = net(images);
	
	ofstream out("features_50.bin",std::ios_base::binary);
	if(!out.good())
	{
		cout << "Error" << endl;
	}
	for (int i = 0; i < 6; ++i) 
	{
		for (int j = 0; j < 128; ++j) {
			out << features[i](0, j) << " ";
		}
	}
	out.close();
	
    std::vector<string> labels;
	labels.push_back("chiungi");
	labels.push_back("chiungi");
	labels.push_back("chiungi");
	labels.push_back("tzuting");
	labels.push_back("tzuting");
	labels.push_back("tzuting");
	
	
    // Run all the images through the network to get their vector embeddings.
	images.clear();
	load_image(image, argv[1]);
    for (auto face : detector(image))
    {
        auto shape = sp(image, face);
        matrix<rgb_pixel> face_chip;
        extract_image_chip(image, get_face_chip_details(shape,50,0.25), face_chip);
        images.push_back(move(face_chip));
    }

    std::vector<matrix<float,0,1>> embedded = net(images);
	
	// print results
	for  (size_t i = 0; i < embedded.size(); ++i)
	{
		cout << i << " ";
		if (length(embedded[i]-features[0]) < net.loss_details().get_distance_threshold())
			cout << labels[0] << endl;
		else if (length(embedded[i]-features[1]) < net.loss_details().get_distance_threshold())
			cout << labels[1] << endl;
		else
			cout << "unknwon" << endl;
	}
}




#include <iostream>
#include <dlib/matrix.h>
#include <fstream>

using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

int main()
{
	ifstream in("test.bin",std::ios_base::binary);
	if(!in.good())
	{
		cout << "Error" << endl;
	}
	
	float f;
	dlib::array<float> float_array = dlib::array<float>();
	
	for (long i = 0; i < 128; ++i) {
		in >> f;
		float_array.push_back(f);
	}
	
	auto float_mat = mat(float_array);
	auto float_matrix = matrix<float, 0, 1>(float_mat);
	
	for (long i = 0; i < 128; ++i) {
		cout << float_matrix(0, i) << "\t";
	}
	
	/*
	ofstream out("test.bin",std::ios_base::binary);
	if(!out.good())
	{
		cout << "Error" << endl;
	}
	
	for (long i = 0; i < 128; ++i) {
		cout << float_matrix(0, i) << "\t";
		out << float_matrix(0, i) << " ";
		//out.write((char *)&f1,sizeof(float));
	}
	*/
	
	cout << endl;
	
	//out.close();
}
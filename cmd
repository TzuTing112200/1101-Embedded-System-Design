- install opencv
https://blog.gtwang.org/programming/ubuntu-linux-install-opencv-cpp-python-hello-world-tutorial/


- install dlib
https://www.twblogs.net/a/5c13ecbdbd9eee5e41840036


- install libx11-dev
https://zoomadmin.com/HowToInstall/UbuntuPackage/libx11-dev


- install libjpeg
https://stackoverflow.com/questions/11969000/how-to-install-libjpeg


- compile for embedded system
https://www.daimajiaoliu.com/daima/47e20dfdb9003fe?fbclid=IwAR2i3mTfihbG3ZgxCJCPjUEd_n4YDQETo7WqwA0Vk2MGb2ibuSSfrSbAGrY




- compile and run in ubuntu

g++ -std=c++11 -O3 -I /home/es/workspace/dlib-19.22/ /home/es/workspace/dlib-19.22/dlib/all/source.cpp -lpthread -lX11 ./dlib_test.cpp -DDLIB_JPEG_SUPPORT -ljpeg -o dlib_test.out
./dlib_test.out /home/es/workspace/Source/shape_predictor_68_face_landmarks.dat /home/es/workspace/Source/test.jpg


g++ -std=c++11 -O3 -I /home/es/workspace/dlib-19.22/ /home/es/workspace/dlib-19.22/dlib/all/source.cpp -lpthread -lX11 ./webcam.cpp -o webcam.out `pkg-config --cflags --libs opencv`
./webcam.out

g++ -std=c++11 -O3 -I /home/es/workspace/dlib-19.22/ /home/es/workspace/dlib-19.22/dlib/all/source.cpp -lpthread -lX11 ./webcam_small.cpp -o webcam_small.out `pkg-config --cflags --libs opencv`
./webcam_small.out

g++ -std=c++11 -O3 -I /home/es/workspace/dlib-19.22/ /home/es/workspace/dlib-19.22/dlib/all/source.cpp -lpthread -lX11 ./webcam_small_50.cpp -o webcam_small_50.out `pkg-config --cflags --libs opencv`
./webcam_small_50.out


g++ -std=c++11 -O3 -I /home/es/workspace/dlib-19.22/ /home/es/workspace/dlib-19.22/dlib/all/source.cpp -lpthread -lX11 ./metrix_test.cpp -o metrix_test.out `pkg-config --cflags --libs opencv`
./metrix_test.out


g++ -std=c++11 -O3 -I /home/es/workspace/dlib-19.22/ /home/es/workspace/dlib-19.22/dlib/all/source.cpp -lpthread -lX11 ./train.cpp -DDLIB_JPEG_SUPPORT -ljpeg -o train.out
./train.out /home/es/workspace/dlib-19.22/examples/johns
./train.out /home/es/workspace/ES/Final/code/images

g++ -std=c++11 -O3 -I /home/es/workspace/dlib-19.22/ /home/es/workspace/dlib-19.22/dlib/all/source.cpp -lpthread -lX11 ./train_small.cpp -DDLIB_JPEG_SUPPORT -ljpeg -o train_small.out
./train_small.out /home/es/workspace/ES/Final/code/images_small

g++ -std=c++11 -O3 -I /home/es/workspace/dlib-19.22/ /home/es/workspace/dlib-19.22/dlib/all/source.cpp -lpthread -lX11 ./train_small_50.cpp -DDLIB_JPEG_SUPPORT -ljpeg -o train_small_50.out
./train_small_50.out /home/es/workspace/ES/Final/code/images_50


g++ -std=c++11 -O3 -I /home/es/workspace/dlib-19.22/ /home/es/workspace/dlib-19.22/dlib/all/source.cpp -lpthread -lX11 ./clip_test.cpp -DDLIB_JPEG_SUPPORT -ljpeg -o clip_test.out
./clip_test.out /home/es/workspace/Source/test.jpg


g++ -std=c++11 -O3 -I /home/es/workspace/dlib-19.22/ /home/es/workspace/dlib-19.22/dlib/all/source.cpp -lpthread -lX11 ./test.cpp -DDLIB_JPEG_SUPPORT -ljpeg -o test.out
./test.out /home/es/workspace/Source/test.jpg

g++ -std=c++11 -O3 -I /home/es/workspace/dlib-19.22/ /home/es/workspace/dlib-19.22/dlib/all/source.cpp -lpthread -lX11 ./test_small.cpp -DDLIB_JPEG_SUPPORT -ljpeg -o test_small.out
./test_small.out /home/es/workspace/Source/test.jpg

g++ -std=c++11 -O3 -I /home/es/workspace/dlib-19.22/ /home/es/workspace/dlib-19.22/dlib/all/source.cpp -lpthread -lX11 ./test_small_50.cpp -DDLIB_JPEG_SUPPORT -ljpeg -o test_small_50.out
./test_small_50.out /home/es/workspace/Source/test.jpg

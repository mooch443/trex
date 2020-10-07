cmake .. -DCMAKE_OSX_DEPLOYMENT_TARGET=10.14 -DCMAKE_MODULE_PATH=/usr/local/Cellar/opencv3/$(ls -t /usr/local/Cellar/opencv3/ | head -1)/share/OpenCV/ -DOpenCV_DIR=/usr/local/Cellar/opencv3/$(ls -t /usr/local/Cellar/opencv3/ | head -1)/share/OpenCV/ -DQTDIR=/usr/local/Cellar/qt5/5.6.0/ -G Xcode
xcodebuild -configuration Release

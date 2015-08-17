rm binary
#g++ binarization.cpp commonUtils/binarization/GraphcutSeg.cpp commonUtils/maxflow/graph.cpp commonUtils/maxflow/maxflow.cpp -lopencv_core -lopencv_highgui -lopencv_imgproc -o binary -w
g++ hog.cpp -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -o binary -w
./binary $1

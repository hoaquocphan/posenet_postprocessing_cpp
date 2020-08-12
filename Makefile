WORK_DIR := ${CURDIR}
#onnxruntime: PoseNetDemo.cpp
onnxruntime: postprocessing.cpp
	${CXX} -std=c++14 postprocessing.cpp \
	-DONNX_ML \
	-I /data/jpeglib/ \
	-I /media/sf_Ubuntu/posenet/posenet_demo/ \
	-L /data/onnxruntime_1.0.0/build/Linux/RelWithDebInfo/ \
	-L /data/onnxruntime_1.0.0/build/Linux/RelWithDebInfo/onnx/ \
	-L /data/onnxruntime_1.0.0/build/Linux/RelWithDebInfo/external/protobuf/cmake/ \
	-L /data/onnxruntime_1.0.0/build/Linux/RelWithDebInfo/external/re2/ \
	-L /data/onnxruntime_1.0.0/cmake/ \
	-L /usr/lib/x86_64-linux-gnu/ \
	-lonnxruntime_session \
	-lonnxruntime_providers \
	-lautoml_featurizers \
	-lonnxruntime_framework \
	-lonnxruntime_optimizer \
	-lonnxruntime_graph \
	-lonnxruntime_common \
	-lonnx_proto \
	-lprotobuf \
	-lre2 \
	-lonnxruntime_util \
	-lonnxruntime_mlas \
	-lonnx \
	-ljpeg -ltbb -ltiff -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_photo -lopencv_imgcodecs \
	-lpthread -O2 -fopenmp -ldl ${LDFLAGS} -o postprocessing

clean:
	rm -rf *.o postprocessing 

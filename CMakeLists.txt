CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)


SET(OpenCV_STATIC OFF)
FIND_PACKAGE(OpenCV REQUIRED)

PROJECT(TP2)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
ADD_EXECUTABLE(canny canny.cpp)
TARGET_LINK_LIBRARIES(canny ${OpenCV_LIBS})

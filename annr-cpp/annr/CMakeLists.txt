set(SOURCES algo/ActiveInterpolator.cpp algo/ActiveInterpolator.h algo/VoronoiGraph.cpp algo/VoronoiGraph.h
        algo/kernels.cpp algo/kernels.h utils.cpp utils.h RandomEngine.cpp RandomEngine.h
        IndexSet.cpp IndexSet.h KDTree.cpp KDTree.h)

link_libraries(cnpy)

add_library(annr MODULE annr_python.cpp ${SOURCES})
target_link_libraries(annr ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
target_include_directories(annr PRIVATE ${PYTHON_INCLUDE_DIRS})

add_executable(annr_test test/annr_test.cpp ${SOURCES})
target_link_libraries(annr_test ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
target_include_directories(annr_test PRIVATE ${PYTHON_INCLUDE_DIRS})
set_target_properties(annr_test PROPERTIES EXCLUDE_FROM_ALL TRUE)

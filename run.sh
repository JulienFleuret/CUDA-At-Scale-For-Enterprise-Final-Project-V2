if [ ! -d build ]; then
mkdir build
fi

cd build && cmake .. -DUNIT_TEST=ON && make -j && ctest -R run_unit_test

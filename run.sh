if [ ! -d build ]; then
mkdir build
fi

cd build

if [ $# -eq 0 ]
  then
	cmake .. -DUNIT_TEST=ON && make -j && ctest -R run_unit_test -V
fi





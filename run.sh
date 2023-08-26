if [ ! -d build ]; then
mkdir build
fi

cd build

if [ $# -eq 0 ]
  then
	make -j && ctest -R run_unit_test -V && cmake .. -DUNIT_TEST=ON
fi





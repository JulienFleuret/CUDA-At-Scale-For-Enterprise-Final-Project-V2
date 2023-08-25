if [ ! -d build ]; then
mkdir build
fi

cd build

if [ $# -eq 0 ]
  then
    cmake .. -DUNIT_TEST=ON
  else
    cmake .. -DUNIT_TEST=ON $1
fi

make -j && ctest -R run_unit_test -V



#!/bin/bash
catkin clean -y
catkin config \
  --cmake-args \
    -DCMAKE_C_COMPILER=/usr/bin/clang \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
    -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" \
    -DCMAKE_CXX_FLAGS="-O3 -ffast-math -march=native -flto -fvectorize -fslp-vectorize"
catkin build

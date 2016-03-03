#!/bin/bash

rm -rf build
mkdir build
cd build

cmake  -G "Unix Makefiles" ..

cd ..
gtags -v


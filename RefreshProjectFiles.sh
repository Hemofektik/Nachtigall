#!/bin/bash

cd build

cmake -G "Unix Makefiles" ..

cd ..
gtags -v


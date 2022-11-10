#!/bin/sh
[ -d build ] || mkdir build
cmake . -B build
cmake --build build

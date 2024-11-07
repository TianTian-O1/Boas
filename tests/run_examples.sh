#!/bin/bash

examples=(
  "print(42)"
  "x = 10\nprint(x)"
  "print(2 + 3 * 5)"
)

for test in "${examples[@]}"; do
  echo "Testing: $test"
  echo "$test" > temp.bs
  ./build/src/boas temp.bs
  echo "-------------------"
done
#!/bin/zsh

for f in cluster/*.vec; do
  ./dbscan.sh cluster ${f:t:r}
done


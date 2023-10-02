#!/bin/bash

DISTANCES=(5 10 20 30)
LRS=(1e-3 1e-4 1e-2)
HID_DIMS=(32)
HEADS=(1 3)
set -e
for distance in "${DISTANCES[@]}"; do
  for lr in "${LRS[@]}"; do
    for hid_dim in "${HID_DIMS[@]}"; do
      for head in "${HEADS[@]}"; do
        python hps_xenium.py --distance $distance --lr $lr --hid_dim $hid_dim --head $head
      done
    done
  done
done
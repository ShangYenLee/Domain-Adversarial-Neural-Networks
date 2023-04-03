#!/bin/bash

case $1 in
    *"svhn"*)
        python inference.py --test_loc $1 --out_path $2 --checkpoint ./checkpoint/ep260_svhn_acc=0.41745.pt;;
    *"usps"*)
        python inference.py --test_loc $1 --out_path $2 --checkpoint ./checkpoint/ep115_usps_acc=0.82392.pt;;
esac
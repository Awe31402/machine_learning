#!/bin/bash
female_file=`ls f*`
male_file=`ls m*`

#echo $female_file
#echo $male_file

for i in $male_file
do
    mv $i test_boy.jpg
    echo "testing $i.."
    octave main.m
    mv test_boy.jpg $i
done

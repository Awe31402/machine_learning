#!/bin/bash
female_file=`ls f*.jpg`
male_file=`ls m*.jpg`

test_gender='test.jpg'
test_files=$male_file
test_files+=" "$female_file

for i in $test_files
do
    mv $i $test_gender
    echo "testing $i.."
    octave main.m
    mv $test_gender $i
done

#!/bin/bash

cd ~/.julia/v0.6/LowRankModels
git pull origin master
git checkout master

cd ~/.julia/v0.6/PyCall
git pull origin master
git checkout master

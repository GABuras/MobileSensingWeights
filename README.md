# MobileSensingWeights

Project 1: Mobile Sensing Exploration for CSC 4562

Uses machine learning to interpret audio picked up by phone to determine the weight on a bar being dropped next to the phone.

USER GUIDE:

py src/otXception/fine_tune.py data/ classes.txt src/otXception/output

To add data, first trim sound of weight hitting the ground to only include the moments just before to just after, then save .wav file to the appropriate folder series in the TrimSounds folder

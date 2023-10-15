# 3D attitude estimation using various filters
Estimate the 3D orientation or attitude from 6 DOF IMU data (accelerometer and gyroscope) using only accelerometer data, only gyroscope data, a complimentary filter, a Madgwick filter, and an Unscented Kalman Filter. (Check the full problem statements here [project1a](https://rbe549.github.io/rbe595/fall2023/proj/p1a/) and [project1b](https://rbe549.github.io/rbe595/fall2023/proj/p1b/))
## Steps to run the code
- Install Numpy, Scipy, and Matplotlib libraries before running the code.
- To run on the first training data in the `Wrapper.py` file in the 'main' function set the variables as:
	IMU_filename = 'imuRaw1' and vicon_filename = 'viconRot1'
- For the other data change the variables accordingly and run the file.
- To generate 3D animations uncomment the specified lines in 'main' function. 
- In Code folder:
  ```
  python Wrapper.py
  ```
## Report
For detailed description of the math see the report [here](Report.pdf).
## Plots and Animations
For the train data 1, plots and animation showing roll, pitch, and yaw for all the filters:
<p float="middle">
<img src="outputs/p1a.png" width="750" height="450"/>
<img src="outputs/p1b.png" width="750" height="450"/>
</p>
<p float="middle">
<img src="outputs/output1.gif" width="750" height="350"/>
</p>

Remaining plots are present in the report and links to rest of the animations are 
[train1](https://www.youtube.com/watch?v=QqZrlZt3IWk), [train2](https://youtu.be/YaMS5Z0NG9c), [train3](https://youtu.be/Bt4ej2pWsNQ), [train4](https://youtu.be/VEVUZr9buow), [train5](https://youtu.be/5XoWXI-sQrE), [train6](https://youtu.be/J3JOtn7tDPE).

## References
1. S. O. H. Madgwick, A. J. L. Harrison and R. Vaidyanathan, "Estimation of IMU and MARG orientation using a gradient descent algorithm," 2011 IEEE International Conference on Rehabilitation Robotics, Zurich, Switzerland, 2011, pp. 1-7, doi: 10.1109/ICORR.2011.5975346.
2. E. Kraft, "A quaternion-based unscented Kalman filter for orientation tracking," Sixth International Conference of Information Fusion, 2003. Proceedings of the, Cairns, QLD, Australia, 2003, pp. 47-54, doi: 10.1109/ICIF.2003.177425.

## Collaborators
Chaitanya Sriram Gaddipati - cgaddipati@wpi.edu

Shiva Surya Lolla - slolla@wpi.edu

Ankit Talele - amtalele@wpi.edu
  

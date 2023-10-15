# Magical Filtering for attitude estimation
Estimate the 3D orientation or attitude from 6 DOF IMU data (accelerometer and gyroscope) using only accelerometer data, only gyroscope data, a complimentary filter, a Madgwick filter, and a Unscented Kalman Filter.
## Steps to run the code
- Install Numpy, Scipy, and Matplotlib libraries before running the code.
- To run on the first training data in the Wrapper.py file in the 'main' function set the variables as:
	IMU_filename = 'imuRaw1' and vicon_filename = 'viconRot1'
- For the other data change the variables accordingly and run the file.
- To generate 3D animations uncomment the specified lines in 'main' function. 
- In Code folder:
  ```
  python Wrapper.py
  ```
  ## Report
  For detailed description of the math see the report [here](Report.pdf).
  

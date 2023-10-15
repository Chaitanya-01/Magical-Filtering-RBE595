## Project 1a:
- Numpy, Matplotlib, and SciPy libraries must be installed before running the code.

- In Code folder the Wrapper.py and helperfuncs.py contains all the necessary functions. For the code to run the data should be stored in the following folder:
        IMU data = Data/Train/IMU/IMU_filename
        Vicon data = Data/Train/Vicon/vicon_filename

- For example if the 1st train set needs to be plotted, in the Wrapper.py file in the 'main' function set the variables as:
        IMU_filename = 'imuRaw1'
        vicon_filename = 'viconRot1'
        For the other training data change the variables accordingly and run the file.
- For test data, keep the data in the following location:
	IMU data = Data/Test/IMU/test_filename
- To run the test data uncomment the corresponding code and comment out the code relevant to accessing the train data as mentioned in the python file. Also to run the test data, comment out all the vicon instances and set 'viconrpy' variable to zero. While running the test data, wherever the initial orientation is used set it to zeros.

- Relative paths from file location are used to load the data.

- Code to generate the videos is commented out and is mentioned in the 'main' function. If videos need to be generated uncomment the code. (It should be noted that the video generation takes more time)
- 'rotplot.py' and IMU params files should be present in the Project folder at the given or appropriate locations.

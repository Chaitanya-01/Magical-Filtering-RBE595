import os
import sys
import time
import math

import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from helperfuncs import *
from rotplot import rotplot

def only_gyro(omega, initial_o,timestamps):
    """
    Performs numerical integration of the IMU gyroscope values and returns the orientations.
 
    Args:
        omega : (3,N) array of angular velocities (in [omega_x, omega_y, omega_z] format each).
        initial_o :(1,3) array of average of the first 200 orientations obtained from vicon ground truth.
        timestamps: (1,N) array of the corresponding IMU time stamps.
 
    Returns:
        orientations: (3,N) array of roll, pitch, and yaw angles (in radians)
    """
    # Creating an empty orientations array 
    _, N = omega.shape 
    orientations = np.zeros((3, N))
    orientations[:,0] = initial_o
    # Numerical integration
    for i in range(N-1):
        # roll (phi), pitch(theta), and yaw (psi)
        phi = orientations[0,i]
        theta = orientations[1,i]
        psi = orientations[2,i]
        # conversion matrix to convert angular velocities (omegas) to rate of change of roll,pitch, and yaw.
        conv_mat = np.array([[1, np.sin(phi)*np.tan(theta),np.cos(phi)*np.tan(theta)],[0, np.cos(phi), -np.sin(phi)],[0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]])
        # rpy_{t+1} = rpy_{t} + rpy_dot*dt
        orientations[:,i+1] = orientations[:,i] + np.dot(conv_mat,omega[:,i])*(timestamps[0,i+1]-timestamps[0,i])
    
    return(orientations)

def only_acc(acc):
    """
    Returns orientations from IMU acceleration data assuming gravity is pointing downwards (-Z).
 
    Args:
        acc : (3,N) array of accelerations (in [a_x, a_y, a_z] format each).
 
    Returns:
        orientations: (3,N) array of roll, pitch, and yaw angles (in radians)
    """
    # Creating an empty orientations array 
    _, N = acc.shape 
    orientations = np.zeros((3, N))
    for i in range(N):
        orientations[:,i] = [np.arctan2(acc[1,i],np.sqrt(acc[0,i]**2 + acc[2,i]**2)),\
                             np.arctan2(-acc[0,i],np.sqrt(acc[1,i]**2 + acc[2,i]**2)),\
                                np.arctan2(np.sqrt(acc[0,i]**2 + acc[1,i]**2),acc[2,i])]

    return(orientations)

def comp_filter(acc,gyro,vicon_rpy,timestamps):
    """
    Returns orientations using complementary filter.
 
    Args:
        acc : (3,N) array of accelerations (in [a_x, a_y, a_z] format each).
        gyro : (3,N) array of angular velocities (in [omega_x, omega_y, omega_z] format each).
        vicon_rpy : (3,N) array of ground truth vicon roll, pitch, and yaw angles.
        timestamps : (1,N) array of the corresponding IMU time stamps.

    Returns:
        orientations: (3,N) array of roll, pitch, and yaw angles (in radians)
    """
    _, N = acc.shape # Get the size of IMU data
    #-------Filtering the data first-------------------------------------------
    # Running a low pass filter on acceleration values
    n = 0.8 #Tunable weight
    acc_filtered = np.zeros((3,N))
    acc_filtered[:,0] = acc[:,0]
    for i in range(N-1):
        acc_filtered[:,i+1] = (1-n)*acc[:,i+1] + n*acc_filtered[:,i]
    
    #Running a high pass filter on gyroscope data
    gyro_filtered = np.zeros((3,N))
    gyro_filtered[:,0] = gyro[:,0]
    for i in range(N-1):
        gyro_filtered[:,i+1] = (1-n)*gyro_filtered[:,i] + (1-n)*(gyro[:,i+1]-gyro[:,i])
     
    #----------Getting orientations from the filtered data------------------------------
    initial_orientation = np.array([np.average(vicon_rpy[0,0:200]), np.average(vicon_rpy[1,0:200]), np.average(vicon_rpy[2,0:200])])
    gyro_orientations = only_gyro(gyro_filtered,initial_orientation,timestamps)
    acc_orientations = only_acc(acc_filtered)
    #--------Fusing the orientations with high and low pass filters-------------------------------
    alpha = 0.8 # Tunable gains
    beta = 0.8
    gamma = 0.9
    # orientations = (1-alpha)*np.array(gyro_orientations) + alpha*np.array(acc_orientations)
    orientations = np.dot(np.array([[1-alpha, 0, 0],[0, 1-beta, 0],[0, 0, 1-gamma]]),np.array(gyro_orientations)) + np.dot(np.array([[alpha, 0, 0],[0, beta, 0],[0, 0, gamma]]),np.array(acc_orientations))
    return orientations

def madg_filter(acc,gyro,vicon_rpy,timestamps):
    """
    Returns orientations using madgwick filter.
 
    Args:
        acc : (3,N) array of accelerations (in [a_x, a_y, a_z] format each).
        gyro : (3,N) array of angular velocities (in [omega_x, omega_y, omega_z] format each).
        vicon_rpy : (3,N) array of ground truth vicon roll, pitch, and yaw angles.
        timestamps : (1,N) array of the corresponding IMU time stamps.

    Returns:
        orientations: (3,N) array of roll, pitch, and yaw angles (in radians)
    """
    _, N = acc.shape # Get the size of IMU data
    #------------Running high and low pass filters---------------------------
    # n1 = 0.01
    # acc_filtered = np.zeros((3,N))
    # acc_filtered[:,0] = acc[:,0]
    # for i in range(N-1):
    #     acc_filtered[:,i+1] = (1-n1)*acc[:,i+1] + n1*acc_filtered[:,i]
    
    # #Running a high pass filter on gyroscope data
    # n2 = 0.01
    # gyro_filtered = np.zeros((3,N))
    # gyro_filtered[:,0] = gyro[:,0]
    # for i in range(N-1):
    #     gyro_filtered[:,i+1] = (1-n2)*gyro_filtered[:,i] + (1-n2)*(gyro[:,i+1]-gyro[:,i])
    # acc = acc_filtered
    # gyro = gyro_filtered
    #------------------------------------------------------------------------
    # First find the initial orientation and from that get the initial quaternion estimate
    initial_orientation = np.array([np.average(vicon_rpy[0,0:200]), np.average(vicon_rpy[1,0:200]), np.average(vicon_rpy[2,0:200])])
    q_init = angle_to_quat(initial_orientation)

    # tuning parameters and initialization
    madg_quat = np.zeros((4,N)) # Initializing an array to store all the estimates
    q_curr = q_init/np.linalg.norm(q_init) # unit initial quaternion qhat_est,t
    madg_quat[:,0] = np.reshape(q_curr,(4,)) 
    beta = 0.1 # Divergence rate
    for i in range(N-1):
        norm_a = np.linalg.norm(acc[:,i+1]) # Normalized acceleration ahat_t+1
        # Calculating the update term
        J = np.array([[-2*q_curr[2,0], 2*q_curr[3,0], -2*q_curr[0,0], 2*q_curr[1,0]],\
                      [2*q_curr[1,0], 2*q_curr[0,0], 2*q_curr[3,0], 2*q_curr[2,0]],\
                        [0, -4*q_curr[1,0], -4*q_curr[2,0], 0]])
        f = np.array([2*(q_curr[1,0]*q_curr[3,0]-q_curr[0,0]*q_curr[2,0])-acc[0,i+1],\
                      2*(q_curr[0,0]*q_curr[1,0]+q_curr[2,0]*q_curr[3,0])-acc[1,i+1],\
                        2*(0.5-q_curr[1,0]**2 -q_curr[2,0]**2)-acc[2,i+1]])
        del_f = np.dot(np.transpose(J),f) # Gradient of f
        update_term = np.reshape(-(beta/np.linalg.norm(del_f))*del_f,(4,1))

        # Calculating the quaternion derivative from gyroscope values
        q_dot_gyro = 0.5*quatn_multiply(q_curr,np.array([0,gyro[0,i+1],gyro[1,i+1],gyro[2,i+1]]))
        # Fusing the quaternions from accelerometer and gyroscope
        q_dot = update_term + q_dot_gyro # qdot_est,t+1 = qdot_omega,t+1 + del_q,t+1 , combining gyro and accelerometer rates
        q_curr = q_curr + q_dot*(timestamps[0,i+1]-timestamps[0,i]) # Next q estimate is found
        q_curr = q_curr/np.linalg.norm(q_curr)
        madg_quat[:,i+1] = np.reshape(q_curr,(4,)) # Adding the new estimate to the array
    
    #--------Converting the quaternions to orientations---------------------------------------------------
    orientations = np.zeros((3,N))
    for i in range(N):
        qw = madg_quat[0,i]
        qx = madg_quat[1,i]
        qy = madg_quat[2,i]
        qz = madg_quat[3,i]
        orientations[0,i] = np.arctan2(2*(qw*qx + qy*qz),1-2*(qx**2 + qy**2))
        orientations[1,i] = np.arcsin(2*(qw*qy-qx*qz))
        orientations[2,i] = np.arctan2(2*(qw*qz + qx*qy),1-2*(qy**2+qz**2))

    return orientations

def ukf(acc,gyro,ts):
    """
    Returns orientations using UKF filter.
 
    Args:
        acc : (3,N) array of accelerations (in [a_x, a_y, a_z] format each).
        omega : (3,N) array of angular velocities (in [omega_x, omega_y, omega_z] format each).
        ts : (1,N) array of the corresponding IMU time stamps.

    Returns:
        orientations: (3,N) array of roll, pitch, and yaw angles (in radians)
    """
    # No. of samples
    _, N_samples = acc.shape

    n = 6 # this creates a (nxn) P matrix which in turn gives 2n sigma points
    #---------------Initialize noise------------------------------------------------------
    # Q = np.diag([100, 100, 100, 0.1, 0.1, 0.1])
    # R = np.diag([1, 1, 1, 0.01, 0.01, 0.01])
    Q = np.diag([106, 106, 106, 0.5, 0.5, 0.5]) # Covariance of the process noise (Tunable)
    R = np.diag([8, 8, 8, 0.1, 0.1, 0.1]) # Covariance of the measurement noise (Tunable)
    #-----------Initialize state vector and covariance-------------------------------------
    x = np.array([1, 0, 0, 0, 0, 0, 0]) # State vector x_k-1
    Pk = np.zeros((6,6)) #1e-2*np.eye(6)# Covariance of state vector P_k-1
    #---------------------------------------------------------------------------------------
    ukf_quats = np.zeros((4,N_samples)) # Array to store all the resulting quaternion estimates
    ukf_quats[:,0] = np.array([1,0,0,0]) # First estimate is added
    #---------------------------------------------------------------------------------------
    for i in range(N_samples-1): # Getting next state estimate through process and measurement update
        del_t = ts[i+1] - ts[i]
        S = np.linalg.cholesky(Pk+Q) # Square root of covariance P
        W = np.concatenate((math.sqrt(n)*S,-math.sqrt(n)*S),axis=1) # Set of disturbances W (6,12) array
        #---------Sigma points--------------------------------------
        X = find_sigma_points(x,W)
        Y = find_tfsigma_points(X,del_t) # transformed sigma points
        #--------Mean of the transformed sigma point distribution--------------------
        # We can't directly average through summation since state vector is not an element of the vector space
        vector_of_errorrots, xbar = intrinsic_grad_descent(X[:,0],Y)
        #--------Covariance of a priori state vector or transformed sigma points Y------------------
        Wdash = find_wdash(vector_of_errorrots,Y,xbar) # New disturbance
        Pk_bar = np.float64(1/float(Wdash.shape[1]))*np.dot(Wdash,Wdash.T) # Covariance
        #--------Getting the set of projected measurement vectors---------------------------
        Z = find_measurementvectZ(Y)
        zk_bar = np.mean(Z,axis=1) # mean of Zi
        #----------Measurement estimate covariance Pzz and Cross correlation matrix Pxz--------------
        temp_var = Z - zk_bar.reshape(-1,1) 
        Pzz = np.float64(1/float(temp_var.shape[1]))*np.dot(temp_var,temp_var.T)
        Pxz = np.float64(1/float(temp_var.shape[1]))*np.dot(Wdash,temp_var.T)
        #--------------------------------------------------------------------------
        vk = np.concatenate((acc[:,i],gyro[:,i]),axis=0) - zk_bar # Innovation vk = actual measurement from data - mean z
        Pvv = Pzz + R # New covariance with noise
        K = np.dot(Pxz,np.linalg.inv(Pvv)) # Kalman gain
        x, Pk= state_update(xbar,K,vk,Pk_bar,Pvv) # new state vector and covariance
        ukf_quats[:,i] = x[0:4].reshape(4,)
    
    #--------Converting the quaternions to orientations-----------------------
    orientations = np.zeros((3,N_samples))
    for i in range(N_samples):
        qw = ukf_quats[0,i]
        qx = ukf_quats[1,i]
        qy = ukf_quats[2,i]
        qz = ukf_quats[3,i]
        roll = np.arctan2(2*(qw*qx + qy*qz),1-2*(qx**2 + qy**2))
        sinp = 2*(qw*qy-qx*qz)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)
        yaw = np.arctan2(2*(qw*qz + qx*qy),1-2*(qy**2+qz**2))
        
        orientations[0,i] = roll
        orientations[1,i] = pitch
        orientations[2,i] = yaw
    return orientations


def animate(i,acc_orientations, gyro_orientations, comp_orientations,madg_orientations,ukf_orientations, vicon_mats):
    """
    This function helps in creation of the videos.
     Args:
        i : iterator variable used for animation.FuncAnimation
        acc_orientations : (3,N) array of orientations (in [roll, pitch, yaw] format each) from only acceleration filter.
        gyro_orientations : (3,N) array of orientations (in [roll, pitch, yaw] format each) from only gyroscope filter.
        comp_orientations : (3,N) array of orientations (in [roll, pitch, yaw] format each) from complimentary filter.
        madg_orientations : (3,N) array of orientations (in [roll, pitch, yaw] format each) from madgwick filter.
        ukf_orientations : (3,N) array of orientations (in [roll, pitch, yaw] format each) from UKF filter.
        vicon_mats : (3,3,N) array of rotation matrices of groundtruth data from vicon.
 
    Returns:
        single frame of animation
    """
    acc_mats = angle_to_mat(acc_orientations)
    gyro_mats = angle_to_mat(gyro_orientations)
    comp_mats = angle_to_mat(comp_orientations)
    madg_mats = angle_to_mat(madg_orientations)
    ukf_mats = angle_to_mat(ukf_orientations)
    
    ax1 = plt.subplot(161,projection='3d',title='gyro(i='+str(i)+')',adjustable='datalim')
    ax2 = plt.subplot(162, projection='3d',title='acc('+str(i)+')',adjustable='datalim')
    ax3 = plt.subplot(163, projection='3d',title='CF('+str(i)+')',adjustable='datalim')
    ax4 = plt.subplot(164, projection='3d',title='madg('+str(i)+')',adjustable='datalim')
    ax5 = plt.subplot(165, projection='3d',title='ukf('+str(i)+')',adjustable='datalim')
    ax6 = plt.subplot(166, projection='3d',title='vicon('+str(i)+')',adjustable='datalim')
    
    # Every 10th frame is used
    rotplot(gyro_mats[:,:,10*i],ax1)
    rotplot(acc_mats[:,:,10*i],ax2)
    rotplot(comp_mats[:,:,10*i],ax3)
    rotplot(madg_mats[:,:,10*i],ax4)
    rotplot(ukf_mats[:,:,10*i],ax5)
    rotplot(vicon_mats[:,:,10*i],ax6)
    


def main():
    # Read Data
    IMU_filename = "imuRaw1"
    vicon_filename =  "viconRot1" # Comment this line when using test data
    # Loading the IMU data, parameters and Vicon Groundtruth data
    absolute_path = os.path.dirname(__file__)

    # For train data use the below four lines
    relativepath_IMUdata = "Data/Train/IMU/"+IMU_filename+".mat"
    fullpath_IMUdata = os.path.join(absolute_path,'..', relativepath_IMUdata)
    relativepath_IMUparams = 'IMUParams.mat'
    fullpath_IMUparams = os.path.join(absolute_path,'..',relativepath_IMUparams)

    # For test data use the below four lines (uncomment)
    # relativepath_IMUdata = "Data/Test/IMU/"+IMU_filename+".mat"
    # fullpath_IMUdata = os.path.join(absolute_path,'..', relativepath_IMUdata)
    # relativepath_IMUparams = 'IMUParams.mat'
    # fullpath_IMUparams = os.path.join(absolute_path,'..',relativepath_IMUparams)

    relativepath_vicon = 'Data/Train/Vicon/'+vicon_filename+'.mat' # Comment this and the below line when using test data
    fullpath_vicon = os.path.join(absolute_path,'..',relativepath_vicon)

    IMU_data = io.loadmat(fullpath_IMUdata)
    IMU_params = io.loadmat(fullpath_IMUparams)['IMUParams']
    vicon_data = io.loadmat(fullpath_vicon) # Comment this line when using test data

    # Seperating the timestamps and values for IMU and Vicon data
    IMU_vals = IMU_data['vals'] # Each column represents the vector of six values (along the rows)
    IMU_ts = IMU_data['ts'] # IMU timestamps
    vicon_rotmat = vicon_data['rots'] # ZYX Euler angles rotation matrix. comment the this and the below line when using test data
    vicon_ts = vicon_data['ts'] # Vicon timestamps

    # Converting the data to physical values with units
    # The bias of the gyroscope values is taken as the average of the first 200 gyroscope readings
    bg = np.array([np.average(IMU_vals[3,0:200]), np.average(IMU_vals[4,0:200]), np.average(IMU_vals[5,0:200])]) # Gyroscope bias [bgx, bgy, bgz]
    # For acceleration values: a_conv = (scale*(a_old) + bias)*9.81 (m/s2)
    # For angular velocities: omega_conv = (3300/1023)*(pi/180)*(0.3)*(omega_old - bias_gyro) (rad/s)
    IMU_vals_converted = np.array([(IMU_vals[0,:]*IMU_params[0,0]+IMU_params[1,0])*9.81,\
        (IMU_vals[1,:]*IMU_params[0,1]+IMU_params[1,1])*9.81,\
            (IMU_vals[2,:]*IMU_params[0,2]+IMU_params[1,2])*9.81,\
                (3300/1023)*(np.pi/180)*0.3*(IMU_vals[3,:]-bg[0]),\
                    (3300/1023)*(np.pi/180)*0.3*(IMU_vals[4,:]-bg[1]),\
                        (3300/1023)*(np.pi/180)*0.3*(IMU_vals[5,:]-bg[2])])
    
    
    vicon_rpy = mat_to_angle(vicon_rotmat) # Converting rotation matrices to roll,pitch,yaw
    # vicon_rpy = np.zeros((3,201)) # uncomment this line when testing and comment the above line

    #--------Getting the orientations using different methods------------------------------------------------------------------------------
    # 1. ONLY GYROSCOPE DATA
    # initial value for numerical integration is obtained as approximation by averaging first 200 values of groundtruth vicon data
    initial_orientation = np.array([np.average(vicon_rpy[0,0:200]), np.average(vicon_rpy[1,0:200]), np.average(vicon_rpy[2,0:200])])  
    omega_zxy = IMU_vals_converted[3:6,:]
    omega_xyz = np.array([omega_zxy[1,:],omega_zxy[2,:],omega_zxy[0,:]])
    gyro_orientaion = only_gyro(omega_xyz,initial_orientation,IMU_ts)

    # 2. ONLY ACCELERATION DATA
    acc_orientation = only_acc(IMU_vals_converted[0:3,:])

    # 3. COMPLIMENTARY FILTER
    comp_orientation = comp_filter(IMU_vals_converted[0:3,:],omega_xyz,vicon_rpy,IMU_ts)

    # 4. MADGWICK FILTER
    madg_orientation = madg_filter(IMU_vals_converted[0:3,:],omega_xyz,vicon_rpy,IMU_ts)
    
    # 5. UKF FILTER
    ukf_orientations = ukf(IMU_vals_converted[0:3,:],omega_xyz,IMU_ts[0,:])

    #---------Plotting the orientations obtained from different methods----------------------------------------------------------------------
    fig = plt.figure(figsize=(10,10))
    
    ax1 = fig.add_subplot()
    ax1 = plt.subplot(3,1,1, title = 'Roll (X)')
    ax1.plot(vicon_ts[0], vicon_rpy[0,:], label ='vicon') # Comment out this line when using test data
    ax1.plot(IMU_ts[0], gyro_orientaion[0,:], label ='gyro')
    ax1.plot(IMU_ts[0], acc_orientation[0,:], label ='acc')
    ax1.plot(IMU_ts[0],comp_orientation[0,:], label ='comp')
    ax1.plot(IMU_ts[0],madg_orientation[0,:], label ='madg')
    ax1.plot(IMU_ts[0],ukf_orientations[0,:], label ='ukf')
    plt.xlabel("timesteps")
    plt.ylabel("angles (rad)")
    plt.legend(loc='upper right')
    ax2 = fig.add_subplot()
    ax2 = plt.subplot(3,1,2, title = 'Pitch (Y)')
    ax2.plot(vicon_ts[0], vicon_rpy[1,:], label ='vicon') # Comment out this line when using test data
    ax2.plot(IMU_ts[0], gyro_orientaion[1,:], label ='gyro')
    ax2.plot(IMU_ts[0], acc_orientation[1,:], label ='acc')
    ax2.plot(IMU_ts[0],comp_orientation[1,:], label ='comp')
    ax2.plot(IMU_ts[0],madg_orientation[1,:], label ='madg')
    ax2.plot(IMU_ts[0],ukf_orientations[1,:], label ='ukf')
    plt.xlabel("timesteps")
    plt.ylabel("angles (rad)")
    plt.legend(loc='upper right')
    ax3 = fig.add_subplot()
    ax3 = plt.subplot(3,1,3, title = 'Yaw (Z)')
    ax3.plot(vicon_ts[0], vicon_rpy[2,:], label ='vicon') # Comment out this line when using test data
    ax3.plot(IMU_ts[0], gyro_orientaion[2,:], label ='gyro')
    ax3.plot(IMU_ts[0], acc_orientation[2,:], label ='acc')
    ax3.plot(IMU_ts[0],comp_orientation[2,:], label ='comp')
    ax3.plot(IMU_ts[0],madg_orientation[2,:], label ='madg')
    ax3.plot(IMU_ts[0],ukf_orientations[2,:], label ='ukf')
    fig.tight_layout()
    plt.xlabel("timesteps")
    plt.ylabel("angles (rad)")
    plt.legend(loc='upper right')
    plt.show()

    #------------Creating videos/Animation (Uncomment below section when using)-------------------------------
    # output_filename = 'output1'
    # _,N_vicon = vicon_rpy.shape
    # ani = animation.FuncAnimation(plt.gcf(), animate, frames=520,fargs=(acc_orientation,gyro_orientaion, comp_orientation,madg_orientation,ukf_orientations,vicon_rotmat), repeat = False)
    # writervideo = animation.FFMpegWriter(fps=60)
    # ani.save(os.path.join(absolute_path,'..', 'outputs/'+output_filename+'.mp4'), writer = writervideo)
    # # plt.tight_layout()
    # # plt.show()


if __name__ == '__main__':
    main()
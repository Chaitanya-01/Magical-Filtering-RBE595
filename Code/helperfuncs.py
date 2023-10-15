import math
import numpy as np


#-------- General Helper Functions------------------------------------
def mat_to_angle(vicon_data):
    """
    Converts the rotation matrices (ZYX) from vicon data to roll pitch and yaw.
 
    Args:
        vicon_data : (3,3,N) array of vicon groundtruth data.
 
    Returns:
        orientations: (3,N) array of roll, pitch, and yaw angles (in radians)
    """
    # Create an empty orientations array
    _,_, N = vicon_data.shape
    orientations = np.zeros((3, N))
    # Convert ZYX mat to rpy angles
    for i in range(N):
        matrix = vicon_data[:,:,i]
        r11, r12, r13 = matrix[0]
        r21, r22, r23 = matrix[1]
        r31, r32, r33 = matrix[2]
        if r11 and r21 ==0:
            orientations[:,i] = [np.arctan2(r12,r22), np.pi/2, 0]
        else:
            orientations[:,i] = [np.arctan2(r32, r33), np.arctan2(-r31, np.sqrt(r11**2 + r21**2)), np.arctan2(r21,r11)]
    return(orientations)

def angle_to_quat(angle):
    """
    Converts the Euler angles to quaternions [qw, qx, qy, qz].
 
    Args:
        angles : (3,1) array of roll, pitch, and yaw.
 
    Returns:
        matrices: (4,1) array of quaternion.
    """
    cr = np.cos(angle[0]*0.5)
    sr = np.sin(angle[0]*0.5)
    cp = np.cos(angle[1]*0.5)
    sp = np.sin(angle[1]*0.5)
    cy = np.cos(angle[2]*0.5)
    sy = np.sin(angle[2]*0.5)
    quat = np.zeros((4,1))
    quat[0] = cr * cp * cy + sr * sp * sy
    quat[1] = sr * cp * cy - cr * sp * sy
    quat[2] = cr * sp * cy + sr * cp * sy
    quat[3] = cr * cp * sy - sr * sp * cy
    return quat

def quatn_multiply(q1,q2):
    """
    Performs quaternion multiplication.
 
    Args:
        q1 and q2 : (4,1) arrays.
 
    Returns:
        product: (4,1) array of quaternion.
    """
    q1 = np.float64(q1)
    q2 = np.float64(q2)
    product = np.zeros((4,1),dtype=np.float64)
    product[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
    product[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
    product[2] = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
    product[3] = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
    return product

def angle_to_mat(angles):
    """
    Converts the Euler angles to Rotation matrix (ZYX).
 
    Args:
        angles : (3,N) array of roll, pitch, and yaw.
 
    Returns:
        matrices: (3,3,N) array of rotation matrices.
    """
    _,N = angles.shape
    matrices = np.zeros((3,3,N))

    for i in range(N):
        roll = angles[0,i]
        pitch = angles[1,i]
        yaw = angles[2,i]
        mat1 = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        mat2 = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        mat3 = np.array([[1, 0, 0],[0, np.cos(roll), -np.sin(roll)],[0, np.sin(roll), np.cos(roll)]])
        matrices[:,:,i] = np.dot(np.dot(mat1,mat2),mat3)
    return matrices

def quat_inv(q):
    """
    Performs quaternion inverse.
    Args:
        q : (4,1) array.
 
    Returns:
        q inverse: (4,) array of quaternion.
    """
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    return q_conj/np.linalg.norm(q)

def omega_to_quat(omega,del_t):
    """
    A function to convert angular velocity or rotation vector into a quaternion through axis angle representation
    Args:
        When del_t = 1:
            omega : (3,1) array representing a rotation vector.
        Otherwise: (i.e. del_t is the difference between two consecutive timesteps)
            omega : (3,1) array representing a angular velocity.
 
    Returns:
        quaternion: (4,) array of quaternion.
    """
    norm = np.linalg.norm(omega)
    if norm == 0:
        q = np.array([1,0,0,0], dtype=np.float64)
        return q
    axis = omega/norm
    q = np.array([math.cos(norm*del_t/2), axis[0]*math.sin(norm*del_t/2), axis[1]*math.sin(norm*del_t/2), axis[2]*math.sin(norm*del_t/2)])
    q = q/np.linalg.norm(q)
    return q

def quat_to_rotvec(q):
    """
    A function to convert quaternion into a rotation vector
    Args:
        q: (4,1) quaternion
 
    Returns:
        (3,) array of rotation vector
    """
    # axis and angle alpha are calculated to find the vector
    sin_alpha_bytwo = np.linalg.norm(q[1:4])
    alpha_bytwo = np.arctan2(sin_alpha_bytwo,q[0])
    alpha = 2*alpha_bytwo
    if sin_alpha_bytwo == 0:
        rot_vec = np.array([0, 0, 0], dtype=np.float64)
        return rot_vec
    axis = q[1:4]/float(sin_alpha_bytwo)
    axis = axis.reshape(3,)
    rot_vec = alpha*axis
    return rot_vec
#-------------UKF specific helper functions-------------------------------------
def find_sigma_points(x,W):
    """
    Computes sigma points (X).
    Args:
        x : state vector (7,) array
        W : set of disturbances (6,12) array.
 
    Returns:
        (7,12) array of sigma points (X)
    """
    _,size = W.shape
    X_top = np.zeros((4,size)) # top part consists of 4 values of the quaternion part
    X_bottom = W[3:6] + x[4:7].reshape(-1,1) # Bottom part has 3 componenets corresponding to the angular velocity or omega
    for i in range(size):
        q_w = omega_to_quat(W[:4,i],1.0)
        X_top[:,i] = quatn_multiply(x[0:4],q_w).reshape(4,)
    
    return np.concatenate((X_top,X_bottom),axis=0)

def find_tfsigma_points(X,del_t):
    """
    Computes sigma points (X).
    Args:
        X : state vector (7,12) array of sigma points
        del_t: the timestep from k-1th step to kth step
 
    Returns:
        (7,12) array of transformed sigma points (Y)
    """
    _,size = X.shape
    Y_top = np.zeros((4,size)) # Yi = A(Xi,0) = [qXi*q_del; omega_Xi]
    Y_bottom = X[4:7,:]
    for i in range(size):
        del_q = omega_to_quat(X[4:7,i],del_t)
        Y_top[:,i] = quatn_multiply(X[:4,i],del_q).reshape(4,)
    
    return np.concatenate((Y_top,Y_bottom), axis=0)

def intrinsic_grad_descent(first_sigmapoint,tfsigma_points):
    """
    Computes mean of the state vector through transformed sigma points.
    Args:
        first_sigmapoint : first sigma point (7,) X1
        tfsigma_points: transformed sigma points (7,12) 
 
    Returns:
        (3,12) array of error rotation vectors (ebar), (7,) array of mean state vector (x_hat,bar)
    """
    _,size = tfsigma_points.shape
    q_bar = first_sigmapoint # Mean orientation is initialized as X1
    threshold = 1e-5
    max_itr = 1000
    curr_iter = 0
    vector_of_errorquats = np.zeros((4,size))
    vector_of_errorrots = np.zeros((3,size))
    mean_error = np.array([100., 100., 100.]) # e_bar

    while(curr_iter < max_itr and np.linalg.norm(mean_error) > threshold):
        for i in range(size):
            ei = quatn_multiply(tfsigma_points[:4,i],quat_inv(q_bar)) #quaternion of shape (4,)
            vector_of_errorquats[:,i] = ei.reshape(4,)
            # Converting the error quaternions to rotation vectors
            ei_bar = quat_to_rotvec(ei) # rotation vector (3,)
            vector_of_errorrots[:,i] = ei_bar # This term is used later while calculating the new covariance of Y
        mean_error = np.mean(vector_of_errorrots,axis=1)
        e = omega_to_quat(mean_error,1)

        q_bar = quatn_multiply(e,q_bar)
        q_bar = np.divide(q_bar, np.linalg.norm(q_bar))
        curr_iter = curr_iter + 1
    omega_bar = np.mean(tfsigma_points[4:7,:],axis = 1)
    x_bar = np.concatenate((q_bar.reshape(4,),omega_bar),axis=0) # shape is (7,)
    return vector_of_errorrots, x_bar

def find_wdash(vect_of_errorrots,tf_sigmapoints,x_bar):
    """
    Computes new disturbance values W' for transformed sigma points.
    Args:
        vect_of_errorrots : (3,12) array of error rotation vectors from previous grad descent
        tfsigma_points: transformed sigma points (7,12)
        x_bar: (7,) mean state vector from gradient descent 
 
    Returns:
        (6,12) array of W'
    """
    # _,size = tf_sigmapoints.shape
    Wdash_top = vect_of_errorrots
    omega_bar = x_bar[4:7]
    Wdash_bottom = tf_sigmapoints[4:7,:]-omega_bar.reshape(-1,1)
    return np.concatenate((Wdash_top,Wdash_bottom), axis=0)

def find_measurementvectZ(tf_sigmapoints):
    """
    Computes measurement state vectors Zi from transformed sigma points Y.
    Args:
        tfsigma_points: transformed sigma points (7,12)
     
    Returns:
        (6,12) array of measurement vectors Zi
    """
    # Zi = H(Yi,0)
    _,size = tf_sigmapoints.shape
    Z = np.zeros((6,size))
    g_quat = np.array([0, 0, 0, 1],dtype=np.float64)
    for i in range(size): 
        Z[:,i] = np.concatenate((quat_to_rotvec(quatn_multiply(quat_inv(tf_sigmapoints[:4,i]),quatn_multiply(g_quat,tf_sigmapoints[:4,i]))),tf_sigmapoints[4:7,i]))
    return Z

def state_update(xbar,K,vk,Pkbar,Pvv):
    """
    Computes updated state vector and updated covariance.
    Args:
        xbar: (7,) mean state vector
        K: Kalman gain (6,6) array
        vK: Innovation (6,)
        Pkbar: covariance of transformed sigma points Yi of size (6,6)
        Pvv: (6,6) covariance of measurement step with added noise
     
    Returns:
        x_new: Updated state vector (7,) array
        Pk: New covariance matrix (6,6)
    """
    temp_var = np.dot(K,vk)
    x_top = quatn_multiply(xbar[:4],omega_to_quat(temp_var[:3],1))
    x_bottom = xbar[4:7] + temp_var[3:6]
    x_new = np.concatenate([x_top.reshape(4,),x_bottom],axis=0)
    Pk = Pkbar - np.dot(K,np.dot(Pvv,K.T))
    return x_new, Pk

    




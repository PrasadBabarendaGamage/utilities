import numpy as np

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def cart2sph(x,y,z):
    # Implemented using same convention as Matlab
    azimuth = np.arctan2(y,x)
    elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r

def sph2cart(azimuth,elevation,r):
    # Implemented using same convention as Matlab
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return x, y, z


def transformRigid3D(x, t):
    """ applies a rigid transform to list of points x.
    T = (tx,ty,tz,rx,ry,rz)
    """
    X = np.vstack((x.T, np.ones(x.shape[0])))
    T = np.array([[1.0, 0.0, 0.0, t[0]],
                  [0.0, 1.0, 0.0, t[1]],
                  [0.0, 0.0, 1.0, t[2]],
                  [1.0, 1.0, 1.0, 1.0]])

    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, np.cos(t[3]), -np.sin(t[3])],
                   [0.0, np.sin(t[3]), np.cos(t[3])]])

    Ry = np.array([[np.cos(t[4]), 0.0, np.sin(t[4])],
                   [0.0, 1.0, 0.0],
                   [-np.sin(t[4]), 0.0, np.cos(t[4])]])

    Rz = np.array([[np.cos(t[5]), -np.sin(t[5]), 0.0],
                   [np.sin(t[5]), np.cos(t[5]), 0.0],
                   [0.0, 0.0, 1.0]])

    T[:3, :3] = np.dot(np.dot(Rx, Ry), Rz)
    return np.dot(T, X)[:3, :].T

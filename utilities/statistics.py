import numpy as np

def uniform_sampling_on_unit_sphere(samples=50):
    theta = 2 * np.pi * np.random.uniform(low=0.0, high=1.0, size=(samples,))
    phi = np.arccos(1 - 2 * np.random.uniform(low=0.0, high=1.0, size=(samples,)))
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    return theta, phi, x, y, z

if __name__ == '__main__':

    theta, phi, x, y, z = uniform_sampling_on_unit_sphere(samples=1000)

    import matplotlib.pyplot as plt
    plt.scatter(np.rad2deg(theta), np.rad2deg(phi))
    plt.xlim(np.rad2deg([0, 2*np.pi]))
    plt.ylim(np.rad2deg([0, np.pi]))
    plt.axis('tight')
    plt.show()
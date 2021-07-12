import pandas as pd
import matplotlib.pyplot as plt
from sklearn import mixture

# Expectation-Maximization clustering algorithm for Hydration III project
# test
# This article explains how the algorithm works:
# https://towardsdatascience.com/implement-expectation-maximization-em-algorithm-in-python-from-scratch-f1278d1b9137
# This example shows how the algorithm works
# https://github.com/VXU1230/Medium-Tutorials/blob/master/em/em.py


# Parameters:
n_clusters = 8 # Defines the number of clusters

# Read drilling data from the csv file
# Columns: time_s cpu_t_degC motor_command active_power_W current_mA arduino_timestamp_ms tacho_rpm imu_x_g imu_y_g imu_z_g WeightOnBit DrillZ1Position_m Formation ROP formation_change
data_unlabeled = pd.read_csv("processed_drilling_data.csv", header = 0)

# Depths and Formation will not be used for clustering, but for graphing and comparing only
depths = data_unlabeled['DrillZ1Position_m']
formation = data_unlabeled['Formation']

# Drop the columns that will not be used. Remaining columns to be used are active_power, current, WeightOnBit, ROP, arduino_timestamp, tacho_rpm, vibration_x, vibration_y, vibration_z
data_unlabeled.drop(['time_s','cpu_t_degC','motor_command','DrillZ1Position_m','Formation', 'formation_change'],axis='columns', inplace=True)
print(data_unlabeled)
print(formation)
print(depths)

# Apply the cluster method Gaussian Mixture to the data: results are cluster centers and cluster standard deviations
gmm = mixture.GaussianMixture(n_components=n_clusters, covariance_type='full')
gmm.fit(data_unlabeled)

# Predict for the data which cluster each data point belongs to
y_pred = gmm.predict(data_unlabeled)
print(y_pred)

# Plot the cluster and the actual formation for each data point as a function of depth
plt.scatter(y_pred, depths, c=[0, 0, 1], marker="x", label="Classification")
plt.plot(formation, depths, c=[1, 0, 0], marker=".", label="Correct")
plt.xlabel("Formation and Classification")
plt.ylabel("Depth")
plt.legend(loc='best')
plt.title("Gaussian Mixture Cluestering")
plt.savefig("Gaussian Mixture Cluestering.png")
plt.clf()
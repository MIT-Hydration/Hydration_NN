# Hydration_NN
Neural Network predictor and clusterng algorithm for Hydration III project

# learning.py

Neural-Network formation prediction for Hydration III project

Input: CSV file containing columns: time_s cpu_t_degC motor_command active_power_W current_mA arduino_timestamp_ms tacho_rpm imu_x_g imu_y_g imu_z_g WeightOnBit DrillZ1Position_m Formation

Output: Plot of the predicted labels and csv input file for clustering


# EMCluster.py

Expectation-Maximization clustering algorithm for Hydration III project

Input: CSV file containing columns: time_s cpu_t_degC motor_command active_power_W current_mA arduino_timestamp_ms tacho_rpm imu_x_g imu_y_g imu_z_g WeightOnBit DrillZ1Position_m Formation ROP formation_change

Output: Plot of the data point clusters

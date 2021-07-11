import torch
import csv
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd

# Neural-Network formation prediction for Hydration III project

# Parameters:
col_depth = 11  # which data column number contains the depth
col_time = 0   # which data column number contains the time
col_ROP = 13 # which data column number contains the Rate of Penetration
col_formation = 12  # which data column number contains the labeled formation
model_vars = [3,4,5,6,7,8,9,10,13] # Column numbers that will be used in the model
model_var_names = ["power", "current","rpm","vib x","vib y","vib z","wob","rop"] # Names of the columns cited above
formations = [0,1,2,3,4,5] # Formation labels. 0 = air, Clay1 = 1, Clay2 = 2, Clay3 = 3, Stone = 4, Concrete = 5
time_range = 20 # number of raw data points to be combined to each model data point
learning_rate = 0.001 # neural-net learning rate
num_iterations = 5000 # number of NN iterations


# Reads the csv data file. Returns the data numbers as an array of floats, and the column names as a vector
def read_input(filename):
    with open(filename,'r') as csvfile:
        names = [] # column names
        read_file = list(csv.reader(csvfile, delimiter=','))

        # Assigns the first file line to the column names
        for j in range(len(read_file[0])):
            names.append(read_file[0][j])
        read_file.pop(0)
        print(names)

        # Reads the numbers to an array of floats
        file_str = list(map(list, zip(*read_file)))
        for i in range(len(file_str)):
            for j in range(len(file_str[i])):
                if file_str[i][j] != '':
                    file_str[i][j] = float(file_str[i][j])
                else:
                    file_str[i][j] = 0
    return file_str, names


# Combines the data points: Each 20 lines become one
def collapse(raw_table, time_range):
    num_lines = int(len(raw_table[0])/time_range)
    num_cols = len(raw_table)
    collapsed_table = []
    for col in range(num_cols): # 13 columns
        temp = []
        # Each line is summed to its time range
        for line in range(num_lines):
            temp.append(sum(raw_table[col][line*time_range:(line+1)*time_range]))
        collapsed_table.append(temp)
    # Each range is divided by number of elements, becoming the average of the elements
    for i in range(len(collapsed_table)):
        for j in range(len(collapsed_table[0])):
            collapsed_table[i][j] /= time_range # potential error: last group may have less than time_range elements
    return collapsed_table

# Calculates the Rate of Penetration of the interval and identifies if formation have changed
def add_rop(collapsed_table):
    rop = [0]
    formation_changed = [0]
    for i in range(len(collapsed_table[0])-1):
        # ROP = delta depth / delta time
        rop.append((collapsed_table[col_depth][i+1]-collapsed_table[col_depth][i])/(collapsed_table[col_time][i+1]-collapsed_table[col_time][i]))
        # formation_changed = 1 if formation has changed, = 0 if formation is the same as previous line
        formation_changed.append(1 - (collapsed_table[col_formation][i+1] == collapsed_table[col_formation][i]))

    collapsed_table.append(rop)
    collapsed_table.append(formation_changed)
    return collapsed_table

# transposes data array
def transpose(raw_table):
    temp = np.array(raw_table)
    temp2 = np.transpose(temp)
    return list(temp2)

# removes data points where bit is moving up (ROP >= 0) and motor or current is off
def filter_drilling(collapsed_table):
    temp = transpose(collapsed_table)
    temp2 = []
    print(len(temp),len(temp[0]))
    for i in range(len(temp)):
        if temp[i][col_ROP] < 0 and temp[i][2] > 0 and temp[i][3] > 0: #column #2 is motor_command, column #3 is active_power_W
            temp2.append(temp[i])
    return transpose(temp2)

# Normalize all columns: new = (old-mean)/StdDev
def normalize(data):
    for i in model_vars:
        temp = np.array(data[i])
        sd = temp.std()
        mean = temp.mean()
        temp = (temp-mean)/sd
        data[i] = list(temp)
    return data

# Plot input data
def plot_data(collapsed_table, names, col_depth = 11):
    for i in range(len(collapsed_table)):
        plt.plot(collapsed_table[i], collapsed_table[col_depth], c=[0, 0, 1])
        plt.xlabel(names[i])
        plt.ylabel(names[col_depth])
        plt.title(names[i] + ' as function of ' + names[col_depth])
        plt.savefig(names[i] + ' as function of ' + names[col_depth] + ".png")
        plt.clf()

# Randomly assign data points to training and testing data
def test_training(data, test_ratio): # test ration between 0 and 1
    temp = transpose(data)
    testing = []
    training = []
    for i in range(len(temp)): # each data point
        if rd.uniform() > test_ratio: # assign to training set
            training.append(temp[i])
        else:
            testing.append(temp[i]) # assign to testing set
    return transpose(training),transpose(testing)

# extract the model data columns from the input data
def get_model_input_data(data):
    model_data = []
    # Columns active_power_W current_mA tacho_rpm imu_x_g imu_y_g imu_z_g WeightOnBit
    model_data.append(data[3]) # power
    model_data.append(data[4]) # current
    model_data.append(data[6]) # rpm
    model_data.append(data[7]) # vib x
    model_data.append(data[8]) # vib y
    model_data.append(data[9]) # vib z
    model_data.append(data[10]) # wob
    model_data.append(data[13])  # rop
    return model_data

# From the Formation column, obtain a boolean column for each formation type
def get_model_output_data(data):
    output = []
    myset = set(data)
    for formation in formations: # for each formation type
        temp = []
        for i in range(len(data)):
            temp.append(formation == data[i]) # adds 1 if data point is related to formation, else adds 0
        output.append(temp)
    return output # returns array where each column is related to a formation type and each line is related to a data point

# Obtains the model probability for each formation, and classifies to the formation with highest probability
def classify_output(output):
    tranp_output = transpose(output)
    maxes = []
    for out_line in tranp_output: # for each data point
        outline = list(out_line)
        maxes.append(outline.index(max(outline))) # returns the formation with highest predicted probability
    return maxes

# Plot result: labeled and predicted data as function of depth
def plot_result(depths, correct, trained):
    for i in range(len(collapsed_table)):
        plt.scatter(trained, depths, c=[0, 0, 1], marker = "x", label = "Classification")
        plt.plot(correct, depths, c=[1, 0, 0], marker = ".", label = "Correct")
        plt.xlabel("Formation")
        plt.ylabel("Depth")
        plt.legend(loc='best')
        plt.title("Formation NN Classification")
        plt.savefig("Formation NN Classification.png")
        plt.clf()



# Model procedure starts here!

# reads drilling data from csv file
# columns: time_s cpu_t_degC motor_command active_power_W current_mA arduino_timestamp_ms tacho_rpm imu_x_g imu_y_g imu_z_g WeightOnBit DrillZ1Position_m Formation
raw_data, col_names = read_input("drill_data_2.csv")

# Combines the data points: Each 20 lines become one
collapsed_table = collapse(raw_data, time_range)

# Calculates the Rate of Penetration of the interval and identifies if formation have changed
collapsed_table = add_rop(collapsed_table)

col_names.append("ROP")
col_names.append("formation_change")
# removes data points where bit is moving up (ROP >= 0) and motor or current is off
drilling_data = filter_drilling(collapsed_table)

# Saves input data to cvs file, to be used for clustering
writer = csv.writer(open("processed_drilling_data.csv", 'w'))
writer.writerow(col_names)
for row in transpose(drilling_data):
    writer.writerow(row)

# Plot input data
#plot_data(drilling_data, col_names)

# Normalize all columns: new = (old-mean)/StdDev
norm_drilling_data = normalize(drilling_data)

# Randomly assign data points to training and testing data
training, testing = test_training(drilling_data, 0.4)
training_input = torch.tensor(transpose(get_model_input_data(training))).float()
testing_input = torch.tensor(transpose(get_model_input_data(testing))).float()

model_output = []

# for each different formation type, train the neural-net using the training set and predict the testing set
for i in formations:
    # From the Formation column, obtain a boolean column for each formation type
    training_output = torch.tensor(get_model_output_data(training[col_formation])[i]).float()

    # Define the NN arquitecture: 2 layers with 12 and 8 nodes, plus sigmoids between layers
    model = nn.Sequential(nn.Linear(8, 12),nn.Linear(12, 8),nn.Sigmoid(),
        nn.Linear(8, 1),nn.Sigmoid(),nn.Flatten(0, 1))

    # Calculate training loss = Mean Square Error
    loss_fn = torch.nn.MSELoss(reduction='sum')

    # repeats backward propagation 5000 times on the training data
    for t in range(num_iterations):
        y_pred = model(training_input) # apply model to training data to predict probability of formation
        loss = loss_fn(y_pred, training_output) # calculate prediction loss
        if t % 1000 == 0:
            print("Iteration: ",t, " Loss: ", loss.item())

        model.zero_grad() # set gradients to zero
        loss.backward() # apply back-propagation to calculate gradient
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad # updates the model with gradient at learning rate

    # apply the model to the testing set
    output = list(model(testing_input).detach().numpy())
    model_output.append(output)

# Obtains the model probability for each formation, and classifies to the formation with highest probability
classification = classify_output(model_output)

# Plot result: labeled and predicted data as function of depth
plot_result(testing[col_depth], testing[col_formation], classification)



import sys
import argparse

parser = argparse.ArgumentParser(description='Hyperparameter tuning logs parser')


parser.add_argument('--file_path', type=str, required=True, help="log file path")
parser.add_argument('--dataset', type=str, required=True, default='ETTm2', help='dataset')
parser.add_argument('--model', type=str, required=True, default='PatchTSMixer', help='Model used in experiment')

args = parser.parse_args()
input_file_name = args.file_path
# Opening log file with path stated in the arguments of the python call
with open(input_file_name, 'r') as file:
    logAsString = file.read()

# Parsing file to get the set of results with corresponding hyperparameters
validation_losses = []
training_losses = []
mse_losses = []
mae_losses = []
hyperparametersValues = []

numOfExperiments = logAsString.count('testing')

experiments_text = logAsString.split('testing')


for exp_index in range(numOfExperiments) :
    last_epoch_validation_loss = experiments_text[exp_index].split('Vali Loss')[-1].split(" ")[1]
    last_epoch_training_loss = experiments_text[exp_index].split('Train Loss')[-1].split(" ")[1]
    last_epoch_mse_loss = experiments_text[exp_index+1].split('mse:')[1].split(",")[0]
    last_epoch_mae_loss = experiments_text[exp_index+1].split('mae:')[1].split("\n")[0]
    validation_losses.append(last_epoch_validation_loss)
    training_losses.append(last_epoch_training_loss)
    mse_losses.append(last_epoch_mse_loss)
    mae_losses.append(last_epoch_mae_loss)
    hyperparametersValues.append(experiments_text[exp_index].split("Namespace")[1].split(")")[0])
    
model = args.model
dataset = args.dataset
horizons = [96, 192, 336, 720]
# Printing best results for each horizon
print("\n\n\n\n------------------ Best results for each horizon with best set of parameters------------------\n\n\n")
print("----------Results of hyperparameter tuning for model " + model + " on dataset " + dataset + "------------\n\n")
for index in range(len(horizons)) :
    minValidationLossIndex = validation_losses[index::len(horizons)].index(min(validation_losses[index::len(horizons)]))
    print("horizon " + str(horizons[index]) + " with hyperparameters : " + hyperparametersValues[index::len(horizons)][minValidationLossIndex] + ")\n")
    print("best validation MSE loss " + validation_losses[index::len(horizons)][minValidationLossIndex] + "\n" + "MSE loss " + mse_losses[index::len(horizons)][minValidationLossIndex] + ", MAE loss " + mae_losses[index::len(horizons)][minValidationLossIndex] + "\n")

    



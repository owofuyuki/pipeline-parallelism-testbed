import numpy as np
import matplotlib.pyplot as plt

# scenario_1
# SPLIT_1_DICT = {'epoch_1': {'accuracy': 4311/10000, 'avg_loss': -1.9556, 'training_time': 467.1616814136505}, 
#                 'epoch_2': {'accuracy': 5487/10000, 'avg_loss': -2.9847, 'training_time': 441.8793895244598}}

# SPLIT_2_DICT = {'epoch_1': {'accuracy': 4731/10000, 'avg_loss': -2.2162, 'training_time': 826.7364029884338}, 
#                 'epoch_2': {'accuracy': 5866/10000, 'avg_loss': -3.4262, 'training_time': 769.2660231590271}}

# SPLIT_4_DICT = {'epoch_1': {'accuracy': 4817/10000, 'avg_loss': -2.2619, 'training_time': 1095.2167716026306}, 
#                 'epoch_2': {'accuracy': 5846/10000, 'avg_loss': -3.2697, 'training_time': 1037.8936145305634}}

# scenario_2
SPLIT_1_DICT = {'epoch_1': {'accuracy': 4812/10000, 'avg_loss': -2.4543, 'training_time': 398.2432375473124}, 
                'epoch_2': {'accuracy': 6028/10000, 'avg_loss': -3.3481, 'training_time': 364.8748967422412}}

SPLIT_2_DICT = {'epoch_1': {'accuracy': 4604/10000, 'avg_loss': -2.2431, 'training_time': 654.728732585907}, 
                'epoch_2': {'accuracy': 5810/10000, 'avg_loss': -3.1167, 'training_time': 624.9075155258179}}

SPLIT_4_DICT = {'epoch_1': {'accuracy': 4293/10000, 'avg_loss': -2.0782, 'training_time': 1048.7231090212412}, 
                'epoch_2': {'accuracy': 5720/10000, 'avg_loss': -3.2869, 'training_time': 992.7812412465213}}

NUM_EPOCH = 2

accuracies_1 = [SPLIT_1_DICT[f"epoch_{i+1}"]["accuracy"] for i in range(NUM_EPOCH)]
avg_losses_1 = [SPLIT_1_DICT[f"epoch_{i+1}"]["avg_loss"] for i in range(NUM_EPOCH)]
training_times_1 = [SPLIT_1_DICT[f"epoch_{i+1}"]["training_time"] for i in range(NUM_EPOCH)]

accuracies_2 = [SPLIT_2_DICT[f"epoch_{i+1}"]["accuracy"] for i in range(NUM_EPOCH)]
avg_losses_2 = [SPLIT_2_DICT[f"epoch_{i+1}"]["avg_loss"] for i in range(NUM_EPOCH)]
training_times_2 = [SPLIT_2_DICT[f"epoch_{i+1}"]["training_time"] for i in range(NUM_EPOCH)]

accuracies_4 = [SPLIT_4_DICT[f"epoch_{i+1}"]["accuracy"] for i in range(NUM_EPOCH)]
avg_losses_4 = [SPLIT_4_DICT[f"epoch_{i+1}"]["avg_loss"] for i in range(NUM_EPOCH)]
training_times_4 = [SPLIT_4_DICT[f"epoch_{i+1}"]["training_time"] for i in range(NUM_EPOCH)]


# Plotting
fig, axs = plt.subplots(1, 3, figsize=(10, 4))

# Plot accuracy
axs[0].plot(range(1, NUM_EPOCH + 1), accuracies_1, marker='.', label = 'split_size = 1', color='cornflowerblue')
axs[0].plot(range(1, NUM_EPOCH + 1), accuracies_2, marker='.', label = 'split_size = 2', color='darkorange')
axs[0].plot(range(1, NUM_EPOCH + 1), accuracies_4, marker='.', label = 'split_size = 4', color='seagreen')
axs[0].set_title('Accuracy over epochs')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Accuracy')
axs[0].grid(True)

# Plot average loss
axs[1].plot(range(1, NUM_EPOCH + 1), avg_losses_1, marker='.', label = 'split_size = 1', color='cornflowerblue')
axs[1].plot(range(1, NUM_EPOCH + 1), avg_losses_2, marker='.', label = 'split_size = 2', color='darkorange')
axs[1].plot(range(1, NUM_EPOCH + 1), avg_losses_4, marker='.', label = 'split_size = 4', color='seagreen')
axs[1].set_title('Average Loss over epochs')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Average Loss')
axs[1].grid(True)

# Plot training time
axs[2].plot(range(1, NUM_EPOCH + 1), training_times_1, marker='.', label = 'split_size = 1', color='cornflowerblue')
axs[2].plot(range(1, NUM_EPOCH + 1), training_times_2, marker='.', label = 'split_size = 2', color='darkorange')
axs[2].plot(range(1, NUM_EPOCH + 1), training_times_4, marker='.', label = 'split_size = 4', color='seagreen')
axs[2].set_title('Training time over epochs')
axs[2].set_xlabel('Epoch')
axs[2].set_ylabel('Training time')
axs[2].grid(True)

plt.tight_layout()
plt.show()
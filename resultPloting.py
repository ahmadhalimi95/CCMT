import os
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib



# Function to load .npy files and calculate average
def load_and_average(directory, file_prefix):
    file_list = [f for f in os.listdir(directory) if f.startswith(file_prefix) and f.endswith('.npy')]
    data = []
    for file_name in file_list:
        data.append(np.load(os.path.join(directory, file_name)))
    
    return np.mean(np.array(data), axis=0)


accuracy_dir = 'accuracy_outputs'
error_dir = 'error_rates'
training_dir = 'training_losses'

# Load and average accuracy_task_1 from all iterations
avg_accuracy_task_1 = load_and_average(accuracy_dir, 'accuracy_task_1_iter')
avg_accuracy_task_1_ohnecu = load_and_average(accuracy_dir, 'accuracy_task_1_ohnecu_iter')
avg_error_task_1 = load_and_average(error_dir,'error_task_1_iter')
avg_error_task_1_ohnecu = load_and_average(error_dir,'error_task_1_ohnecu_iter')
avg_train_loss_su1 = load_and_average(training_dir, 'train_loss_su1_iter')
avg_train_loss_su_ohnecu = load_and_average(training_dir, 'train_loss_su_ohnecu_iter')

avg_accuracy_task_2 = load_and_average(accuracy_dir, 'accuracy_task_2_iter')
avg_error_task_2 = load_and_average(error_dir, 'error_task_2_iter')




save_formats = ['pdf', 'png', 'eps']


#plt.figure(figsize=(10, 5))
#plt.figure()
#plt.plot(self.train_losses, label='System Training Loss')
#plt.title('System Training Loss Convergence')
#plt.xlabel('Epoch')
#plt.ylabel('Loss')
#plt.legend()
#plt.grid(True)
#tikzplotlib.save("Total_loss_plot.tikz")
#for format in save_formats:
#    filename = f'system_training_loss_plot.{format}'
#    plt.savefig(filename, format=format, bbox_inches='tight', dpi=300)
#plt.show()

#combined plot of training losses of specific units
plt.figure()
plt.semilogy(avg_train_loss_su1, label='SU Training Loss (With CU)', color= 'blue', marker='o')
plt.semilogy(avg_train_loss_su_ohnecu, label='SU Training Loss (Without CU)', color= 'red', marker='x')
#plt.title('Training Loss Convergence')
plt.xlabel('Epoch')
plt.ylabel('Losses')
plt.legend()
plt.grid(True)
tikzplotlib.save("Training_loss_plot.tikz")
for format in save_formats:
    filename = f'combined_training_loss_plot.{format}'
    plt.savefig(filename, format=format, bbox_inches='tight', dpi=300)
plt.show()

# Ploting logarithmic error rates
plt.figure()
plt.semilogy(avg_error_task_1, label='Task1 error rate (With CU)', color='blue', alpha=0.2, marker='o')
plt.semilogy(avg_error_task_1_ohnecu, label='Task1 error rate (Without CU)', color='red', alpha=0.2, marker='x')
#plt.semilogy(avg_error_task_2, label='Task2 error rate', color='orange', alpha=0.2, marker='d')
#plt.title('Impact of the CU')
plt.xlabel('Epoch')
plt.ylabel('Tasks execution error rate')
plt.legend()
plt.grid(True)
tikzplotlib.save("Tasks_error_rates_plot.tikz")
for format in save_formats:
    filename = f'combined_error_rates_plot.{format}'
    plt.savefig(filename, format=format, bbox_inches='tight', dpi=300)
plt.show()
        
# Plot of accuricies
plt.figure()
plt.plot(avg_accuracy_task_1, label='Accuracy of Task1 (With CU)', color='blue', alpha=0.2, marker='o')
plt.plot(avg_accuracy_task_1_ohnecu, label='Accuracy of Task1 (Without CU)', color='red', alpha=0.2, marker='x')
#plt.plot(avg_accuracy_task_2, label='Accuracy of Task2', color='orange', alpha=0.2, marker='d')
#plt.title('Accuracy of Task Execution (Impact of CU)')
plt.xlabel('Epoch')
plt.ylabel('Tasks execution accuracy (%)')
plt.legend()
plt.grid(True)
tikzplotlib.save("Tasks_accuracies_plot.tikz")
for format in save_formats:
    filename = f'tasks_accuracies_time.{format}'
    plt.savefig(filename, format=format, bbox_inches='tight', dpi=300)
plt.show()

# Plot of accuricies logarithmic scale
#plt.figure()
#plt.semilogy(avg_accuracy_task_1, label='Accuracy of Task1 (With CU)', color='blue', alpha=0.2, marker='o')
#plt.semilogy(avg_accuracy_task_1_ohnecu, label='Accuracy of Task1 (Without CU)', color='red', alpha=0.2, marker='x')
#plt.title('Accuracy of Task Execution (Impact of CU)')
#plt.xlabel('Epoch')
#plt.ylabel('Tasks execution accuracy')
#plt.legend()
#tikzplotlib.save("log_Tasks_accuracies_plot.tikz")
#for format in save_formats:
#    filename = f'log_tasks_accuracies_time.{format}'
#    plt.savefig(filename, format=format, bbox_inches='tight', dpi=300)
#plt.show()
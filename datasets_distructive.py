import torch
import os
import random
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt



def load_mnist(path="datasets/MNIST", seed=42):
    # set seed
    random.seed(seed)
    np.random.seed(seed)

    # load Dataser
    os.makedirs(path, exist_ok=
                True)
    
    mnist_transform = transforms.Compose([transforms.ToTensor()])
    train = torchvision.datasets.MNIST(root=path, download=True, train=True, transform=mnist_transform)
    test_dataset  = torchvision.datasets.MNIST(root=path, download=True, train=False, transform=mnist_transform)

    #test.data = ...  ->   It should be incorrect if I do test.data = test.data.view() -->  I will do it later in "enumerate" loop

    #train.data = train.data.view(-1, 28*28) / 255 -----> It caused problem here and I guess it was because I affected the original dataset 
    # mnist_data = train  ---- used for the inference part

    # Tasks
    S1_train_targets = (train.targets == 2)
    S2_train_targets = (train.targets == 4)
    # The rest
    S_train_targets  = ~S1_train_targets & ~S2_train_targets


    # Applying the trick to get rid of different subset of dataset
    S1_train_dataset = torch.utils.data.Subset(train, torch.where(S1_train_targets)[0])
    S2_train_dataset = torch.utils.data.Subset(train, torch.where(S2_train_targets)[0])
    S_train_dataset  = torch.utils.data.Subset(train, torch.where(S_train_targets)[0])

    num_samples = 5000

    indices_S1 = np.random.choice(len(S1_train_dataset), num_samples, replace=False)
    indices_S2 = np.random.choice(len(S2_train_dataset), num_samples, replace=False)
    indices_S = np.random.choice(len(S_train_dataset), num_samples, replace=False)


    # Create new datasets with selected samples
    S1_subset = torch.utils.data.Subset(S1_train_dataset, indices_S1)
    S2_subset = torch.utils.data.Subset(S2_train_dataset, indices_S2)
    S_subset = torch.utils.data.Subset(S_train_dataset, indices_S)

    new_dataset = torch.utils.data.ConcatDataset([S1_subset, S2_subset, S_subset])

    # To prevent overfitting I need to shuffle the subsets

    data_loader = torch.utils.data.DataLoader(new_dataset, batch_size=100, shuffle=True)

    # Note: the "generator" in DataLoader helps to have the same shuffling each time we run the program:  ,generator=torch.Generator().manual_seed(seed)

    new_dataset = data_loader.dataset

    # Number of times to repeat the new_dataset to increase the Data for training 
    num_repeats = 3

    # Create a list of repeated datasets
    repeated_datasets = [new_dataset] * num_repeats

    # Concatenate the repeated datasets
    final_dataset = torch.utils.data.ConcatDataset(repeated_datasets)

    # Now I create two training datasets for joint training approach of multitask learning --------------
    # First I should overcome the "atrribute" of targets problem
    samples, targets = zip(*final_dataset)
    samples = torch.stack(samples).view(-1,784) / 255
    targets = torch.tensor(targets)

    # Task1: classification of 2
    task_one_targets = torch.zeros_like(targets)
    task_one_targets[targets == 2] = 1
    first_task_dataset = torch.utils.data.TensorDataset(samples, task_one_targets)

    # Task2: classification of 4
    task_two_targets = torch.zeros_like(targets)
    task_two_targets[targets == 4] = 1
    second_task_dataset = torch.utils.data.TensorDataset(samples, task_two_targets)

    # ---->> Test set of MNIST has a different datatype from the training set, take care
    test_dataset.data = test_dataset.data.to(torch.float32)
    test_dataset.targets = test_dataset.targets.to(torch.float32)

    test_sample = test_dataset.data.view(-1,784) / 255
    # Accuracy Test: Task1
    first_test_target = torch.zeros_like(test_dataset.targets)
    first_test_target[test_dataset.targets == 2] = 1
    first_test_dataset = torch.utils.data.TensorDataset(test_sample, first_test_target)

    # Accuracy Test: Task2
    second_test_target = torch.zeros_like(test_dataset.targets)
    second_test_target[test_dataset.targets == 4] = 1
    second_test_dataset = torch.utils.data.TensorDataset(test_sample, second_test_target)

    return first_task_dataset, second_task_dataset, first_test_dataset, second_test_dataset

# Why the data has been changed to numpy?! Maybe because it is for a long time ago and hasn't been updated

def load_dataset(key):
    loader = {
        "MNIST": load_mnist,
        #"OMNIGLOT": load_omniglot,
        #"Histopathology": load_histopathology,
        #"FreyFaces": load_freyfaces,
        #"OneHot": load_one_hot
    }
    return loader[key]()

 


# Tests and veryfing


#first_task_dataset, second_task_dataset, first_test_dataset, second_test_dataset = load_dataset(key="MNIST")

#---->> (checking the training sets we have made)

# Convert the first_task_dataset to numpy arrays 
#sample_array, target_array = zip(*second_task_dataset)
#sample_array = torch.stack(sample_array).numpy()
#target_array = torch.stack(target_array).numpy()

# Now you can check the shape
#print("CommonInput shape:", sample_array.shape)
#print("Label shape:", target_array.shape)


#num_label_0 = np.sum(target_array == 0)

#print(f"Number of data points labeled 0: {num_label_0}")

# Assuming resampled_dataset is your resampled dataset
#N = len(second_task_dataset)

#print(f"Number of samples in the first task dataset: {N}")


# ---->> (Ploting a recovered data)

#index_4 = next(i for i, (image, label) in enumerate(second_task_dataset) if label == 0)

# Extract and plot the image
#image_4, label = second_task_dataset[index_4]
#image_4 = image_4.numpy()  # Convert to NumPy array
#image_4 = image_4.reshape(28, 28)  # Reshape to the original 2D shape
#plt.imshow(image_4, cmap='gray')
#plt.title(f'MNIST Image labeled as {label}')
#plt.show()


# ---->> (Testing the way we use the test dataset reshaping in the eval loop)

#test_reshaped = test_dataset.data.view(-1,784).squeeze()

#neu_test_dataset = torch.utils.data.TensorDataset(test_reshaped, test_dataset.targets)


#index_4 = next(i for i, (image, label) in enumerate(second_test_dataset) if label == 1)

# Extract and plot the image
#image_4, label = second_test_dataset[index_4]
#image_4 = image_4.numpy()  # Convert to NumPy array
#image_4 = image_4.reshape(28, 28)  # Reshape to the original 2D shape
#plt.imshow(image_4, cmap='gray')
#plt.title(f'MNIST Image labeled as {label}')
#plt.show()

# Assuming your datasets are named first_task_dataset, second_task_dataset, first_test_dataset, and second_test_dataset

# Check data type of training datasets
#print("First Task Training Dataset Data Type:", first_task_dataset.tensors[0].dtype)
#print("Second Task Training Dataset Data Type:", second_task_dataset.tensors[0].dtype)

# Check data type of testing datasets
#print("First Task Test Dataset Data Type:", first_test_dataset.tensors[0].dtype)
#print("Second Task Test Dataset Data Type:", second_test_dataset.tensors[0].dtype)

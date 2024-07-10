import argparse
import os
import random

import numpy as np

from datasets import load_dataset

from CUSUModel import MultiTaskMultiUserComm

# Function to run the training and save results
def run_training(iteration, dataset, lr, seed):
    print(f"Iteration {iteration + 1} - Dataset: {dataset}, Learning Rate: {lr}, Seed: {seed}")

    # Set seed
    random.seed(seed)
    np.random.seed(seed)

    # Get dataset
    first_task_dataset, second_task_dataset, first_test_dataset, second_test_dataset = load_dataset(key=dataset)

    # Train
    save_dir = f"save/{dataset}/"
    save_path = f"save/{dataset}/model_{lr}_{seed}_iter{iteration}"
    os.makedirs(save_dir, exist_ok=True)

    model = MultiTaskMultiUserComm(n_in=784, n_c_latent=64, n_x_latent=16, n_c_h=128, n_x_h=32)

    print(f"Model: {type(model)}")

    model.fit(first_dataset=first_task_dataset, second_dataset=second_task_dataset, first_test_dataset=first_test_dataset,
              second_test_dataset=second_test_dataset, batch_size=100, n_epoch_primal=300, learning_rate=lr, path=save_path)

    # Save numpy files
    os.makedirs("results/", exist_ok=True)
    np.save(f"results/exp_{dataset}_train_loss_{lr}_{seed}_iter{iteration}.npy", np.array(model.train_losses))
    np.save(f"results/exp_{dataset}_train_time_{lr}_{seed}_iter{iteration}.npy", np.array(model.train_times))

    # Saving Accuracy outputs for later Modifications
    accuracy_save_dir = 'accuracy_outputs'
    os.makedirs(accuracy_save_dir, exist_ok=True)
    np.save(os.path.join(accuracy_save_dir, f'accuracy_task_1_iter{iteration}.npy'), np.array(model.accuracy_task_1))
    np.save(os.path.join(accuracy_save_dir, f'accuracy_task_2_iter{iteration}.npy'), np.array(model.accuracy_task_2))
    np.save(os.path.join(accuracy_save_dir, f'accuracy_task_1_ohnecu_iter{iteration}.npy'), np.array(model.accuracy_task_1_ohnecu))

    # Saving loss values and time
    training_save_dir = 'training_losses'
    os.makedirs(training_save_dir, exist_ok=True)
    np.save(os.path.join(training_save_dir, f'train_loss_su1_iter{iteration}.npy'), np.array(model.train_loss_su1))
    np.save(os.path.join(training_save_dir, f'train_loss_su2_iter{iteration}.npy'), np.array(model.train_loss_su2))
    np.save(os.path.join(training_save_dir, f'train_loss_su_ohnecu_iter{iteration}.npy'), np.array(model.train_loss_su_ohnecu))

    # Saving error rates (These are not in percentage)
    errorrate_save_dir = 'error_rates'
    os.makedirs(errorrate_save_dir, exist_ok=True)
    np.save(os.path.join(errorrate_save_dir, f'error_task_1_iter{iteration}.npy'), np.array(model.error_task_1))
    np.save(os.path.join(errorrate_save_dir, f'error_task_2_iter{iteration}.npy'), np.array(model.error_task_2))
    np.save(os.path.join(errorrate_save_dir, f'error_task_1_ohnecu_iter{iteration}.npy'), np.array(model.error_task_1_ohnecu))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Semantic Common Unit with Implicit Optimal Priors.")
    parser.add_argument("--dataset", type=str, default="MNIST", help="Dataset Name.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning Rate for CU.")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed.")

    args = parser.parse_args()

    dataset = args.dataset
    lr = args.learning_rate
    seed = args.seed

    # Run training for 20 iterations
    for i in range(20):
        run_training(i, dataset, lr, seed)

import os
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import tikzplotlib
import numpy as np
from utils import awgn_channel, reparameterize


class MultiTaskNetwork(nn.Module):
    def __init__(self, n_in, n_c_latent, n_c_h, n_x_in, n_x_latent, n_x_h):
        super(MultiTaskNetwork, self).__init__()

        self.n_in = n_in
        self.n_c_latent = n_c_latent
        self.n_h = n_c_h
        self.n_x_in = n_x_in
        self.n_x_latent = n_x_latent
        self.n_x_h = n_x_h

#------ CU-Encoder
        self.cu = nn.Sequential(
            nn.Linear(n_in, n_c_h), nn.Tanh(),
        )
        self.cu_output_mu = nn.Linear(n_c_h, n_c_latent)
        self.cu_output_ln_var = nn.Linear(n_c_h, n_c_latent)

#------ SU1-Encoder
        self.sue_one = nn.Sequential(
            nn.Linear(n_x_in, n_x_h), nn.Tanh(),
        )
        self.sue_one_output = nn.Linear(n_x_h, n_x_latent)

        # SU1-Decoder
        self.sud_one = nn.Sequential(
            nn.Linear(n_x_latent, 16), nn.Tanh(),
        )
        self.sud_one_out = nn.Linear(16, 1)

#------ SU2-Encoder
        self.sue_two = nn.Sequential(
            nn.Linear(n_x_in, n_x_h), nn.Tanh(),
        )
        self.sue_two_output = nn.Linear(n_x_h, n_x_latent)

        # SU2-Decoder
        self.sud_two = nn.Sequential(
            nn.Linear(n_x_latent, 16), nn.Tanh(),
        )
        self.sud_two_out = nn.Linear(16, 10) 

    def cu_encode(self, s):
        h = self.cu(s)
        return self.cu_output_mu(h), self.cu_output_ln_var(h)
    
    def su_t1_encode(self, c):
        h = self.sue_one(c)
        return self.sue_one_output(h)

    def su_t1_decode(self, x):
        h = self.sud_one(x)
        return self.sud_one_out(h)
    
    def su_t2_encode(self, c):
        h = self.sue_two(c)
        return self.sue_two_output(h)

    def su_t2_decode(self, x):
        h = self.sud_two(x)
        return self.sud_two_out(h)
    

class SU_ohne_CUnet(nn.Module):
    def __init__(self, n_x_h, n_x_latent):
        super(SU_ohne_CUnet, self).__init__()

        self.n_x_h = n_x_h
        self.n_x_latent = n_x_latent

        #Encoder
        self.su_ohnecu_enc = nn.Sequential(
            nn.Linear(784, n_x_h), nn.Tanh(),
        )
        self.su_ohnecu_enc_output = nn.Linear(n_x_h, n_x_latent)

        #Decoder
        self.su_ohnecu_dec = nn.Sequential(
            nn.Linear(n_x_latent, 16), nn.Tanh(),
        )
        self.su_ohnecu_dec_out = nn.Linear(16, 1)

    def su_ohnecu_encode(self, s):
        h = self.su_ohnecu_enc(s)
        return self.su_ohnecu_enc_output(h)

    def su_ohnecu_decode(self, x):
        h = self.su_ohnecu_dec(x)
        return self.su_ohnecu_dec_out(h)





class MultiTaskMultiUserComm:

    def __init__(self, n_in, n_c_latent, n_c_h, n_x_latent, n_x_h):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.multitask_network = MultiTaskNetwork(n_in=n_in, n_c_latent= n_c_latent, n_c_h=n_c_h, n_x_in=n_c_latent, n_x_latent=n_x_latent, n_x_h=n_x_h).to(self.device)
        self.SUohneCU_network = SU_ohne_CUnet(n_x_h=n_x_h, n_x_latent=n_x_latent).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.categ_criterion = torch.nn.CrossEntropyLoss() 
        

        self.train_loss_su1 = []
        self.train_loss_su2 = []
        self.train_losses = []
        self.train_times = []
        self.accuracy_task_1 = []
        self.accuracy_task_2 = []
        self.error_task_1 = []
        self.error_task_2 = []

        self.accuracy_task_1_ohnecu = []
        self.error_task_1_ohnecu = []
        self.train_loss_su_ohnecu = []
              


    def _loss_SU1(self, s, targets1):
        c_mu, c_ln_var = self.multitask_network.cu_encode(s)
        c = reparameterize(c_mu, c_ln_var)
        x = self.multitask_network.su_t1_encode(c)
        x_hat = awgn_channel(x)
        z     = self.multitask_network.su_t1_decode(x_hat)
        targets1 = targets1.float()
        sre = self.criterion(z, targets1.view(-1,1))  # This takes care of sigmoid itself and has a negative
        sre_sum = torch.sum(sre) # semantic recovery error(sre)           
        loss_su1 = sre_sum 
        return loss_su1


    def _loss_SU2(self, s, targets2):
        c_mu, c_ln_var = self.multitask_network.cu_encode(s)
        c = reparameterize(c_mu, c_ln_var)
        x2 = self.multitask_network.su_t2_encode(c)
        x2_hat = awgn_channel(x2)
        z2 = self.multitask_network.su_t2_decode(x2_hat)
        sre2 = self.categ_criterion(z2, targets2) 
        sre2_sum = torch.sum(sre2)            
        loss_su2 = sre2_sum 
        return loss_su2
    
    def _loss_SU_ohneCU (self, s, targets1):
        x = self.SUohneCU_network.su_ohnecu_encode(s)
        x_hat = awgn_channel(x)
        z = self.SUohneCU_network.su_ohnecu_decode(x_hat)
        targets1 = targets1.float()
        recovery_error = self.criterion(z, targets1.view(-1,1))
        error_sum = torch.sum(recovery_error)
        loss_su_ohne_cu = error_sum
        return loss_su_ohne_cu

    

    def fit(self, first_dataset, second_dataset, first_test_dataset, second_test_dataset, batch_size=100,
            n_epoch_primal=500, learning_rate=0.01, path=None):

        
        N = 5000    # It must be 15000 if we repeat the dataset                                            

        first_loader = DataLoader(first_dataset, batch_size=batch_size, shuffle=True)
        second_loader = DataLoader(second_dataset, batch_size=batch_size, shuffle=True)
     

        optimizer = torch.optim.Adam(self.multitask_network.parameters(), lr=learning_rate)
        su_ohne_optimizer = torch.optim.Adam(self.SUohneCU_network.parameters(), lr=learning_rate)

        for epoch_primal in range(n_epoch_primal):
            start = time.time()


            mean_loss_su1 = 0
            mean_loss_su2 = 0
            mean_loss = 0

            mean_loss_su_ohne = 0

            zipped_dataloader = zip(first_loader, second_loader)
            self.multitask_network.train()
            self.SUohneCU_network.train()


            for batch, ((S, targets1), (S2, targets2)) in enumerate(zipped_dataloader):
                
                S, targets1, S2, targets2 = S.squeeze().to(self.device), targets1.to(self.device), S2.squeeze().to(self.device), targets2.to(self.device) 
                
                self.multitask_network.zero_grad()

                loss_SU1 = self._loss_SU1(S, targets1=targets1) 
                mean_loss_su1 += loss_SU1.item() / N

                loss_SU2 = self._loss_SU2(S2, targets2=targets2)
                mean_loss_su2 += loss_SU2.item() / N

                loss = loss_SU1 + loss_SU2
                mean_loss = mean_loss_su1 + mean_loss_su2

                loss.backward(retain_graph=True)
                optimizer.step()


            for batch, (S, targets1) in enumerate(first_loader):
                S, targets1 = S.squeeze().to(self.device), targets1.to(self.device)

                self.SUohneCU_network.zero_grad()

                loss_SU_ohne = self._loss_SU_ohneCU(S, targets1)
                mean_loss_su_ohne += loss_SU_ohne.item() / N
                loss_SU_ohne.backward()
                su_ohne_optimizer.step() 

            
            #self.multitask_network.eval()
            test_accuracy_task_one, test_accuracy_task_two, test_accuracy_task_one_ohne, error_rate_task1, error_rate_task2, error_rate_task1_ohnecu = self.eval(first_test_dataset, second_test_dataset)


            end = time.time()
            self.train_loss_su1.append(mean_loss_su1)
            self.train_loss_su2.append(mean_loss_su2)
            self.train_losses.append(mean_loss)
            self.train_times.append(end - start)
            self.accuracy_task_1.append(test_accuracy_task_one)
            self.accuracy_task_2.append(test_accuracy_task_two)
            self.error_task_1.append(error_rate_task1)
            self.error_task_2.append(error_rate_task2)
            self.train_loss_su_ohnecu.append(mean_loss_su_ohne)
            self.accuracy_task_1_ohnecu.append(test_accuracy_task_one_ohne)
            self.error_task_1_ohnecu.append(error_rate_task1_ohnecu)


            print(
                f"VAE epoch: {epoch_primal} / Train: {mean_loss:0.3f} / SU1: {mean_loss_su1:0.3f} / SU2: {mean_loss_su2:0.3f} / SU_ohne: {mean_loss_su_ohne:0.3f}")
            
            print(
                f"Test Accuracy Task One: {test_accuracy_task_one:.2f}% / Test Accuracy Task Two: {test_accuracy_task_two:.2f}% / Test Accuracy Ohne CU: {test_accuracy_task_one_ohne:.2f}%")







    def eval(self, first_test_dataset, second_test_dataset, batch_size= 100, threshold=0.5):

        first_test_loader = DataLoader(first_test_dataset, batch_size=batch_size, shuffle=False)
        second_test_loader = DataLoader(second_test_dataset, batch_size=batch_size, shuffle=False)

        self.multitask_network.eval()
        self.SUohneCU_network.eval()
        
        total_tone = 0
        correct_tone = 0

        total_ttwo = 0
        correct_ttwo = 0

        correct_tone_ohne = 0

        zipped_testloader = zip(first_test_loader, second_test_loader)

        with torch.no_grad():
            for batch ,((first_test_data, first_test_target),(second_test_data, second_test_target)) in enumerate(zipped_testloader):

                first_test_data = first_test_data.squeeze().to(self.device)
                first_test_target = first_test_target.to(self.device)
                second_test_target = second_test_target.to(self.device)

                c_mu, c_ln_var = self.multitask_network.cu_encode(first_test_data)
                c = reparameterize(c_mu, c_ln_var)
                # First Task
                x1 = self.multitask_network.su_t1_encode(c)
                x1_hat = awgn_channel(x1)
                z1 = self.multitask_network.su_t1_decode(x1_hat)
                z1_inferred = torch.sigmoid(z1)
                # Second Task
                x2 = self.multitask_network.su_t2_encode(c)
                x2_hat = awgn_channel(x2)
                z2 = self.multitask_network.su_t2_decode(x2_hat)
                z2_inferred = torch.softmax(z2, dim=1)
                # First aber ohne CU
                x3 = self.SUohneCU_network.su_ohnecu_encode(first_test_data)
                x3_hat = awgn_channel(x3)
                z3 = self.SUohneCU_network.su_ohnecu_decode(x3_hat)
                z3_inferred = torch.sigmoid(z3)

                # Tests
                predicted_t1 = (z1_inferred > threshold).float()
                total_tone += first_test_target.size(0)
                first_test_target = first_test_target.view_as(predicted_t1)  # Ensuring target has the same shape as predicted
                correct_tone += torch.sum((predicted_t1 == first_test_target).float())   

                predicted_t2 = torch.argmax(z2_inferred, dim=1)
                total_ttwo += second_test_target.size(0)
                second_test_target = second_test_target.view_as(predicted_t2)  # Ensuring target has the same shape as predicted
                correct_ttwo += torch.sum((predicted_t2 == second_test_target).float())

                predicted3_t1 = (z3_inferred > threshold).float()
                correct_tone_ohne += torch.sum((predicted3_t1 == first_test_target).float()) 
        
        
        #print('Sum of First predictions:', predicted_t1.sum().item())
        accuracy_task_one = (correct_tone / total_tone) * 100
        error_task_one = 1 - (correct_tone / total_tone)
        #print('Accuracy of the First Task: {:.2f}%'.format(accuracy_task_one))

        #print('Sum of Second predictions:', predicted_t2.sum().item())
        accuracy_task_two = (correct_ttwo / total_ttwo) * 100
        error_task_two = 1 - (correct_ttwo / total_ttwo)
        #print('Accuracy of the Second Task: {:.2f}%'.format(accuracy_task_two))

        accuracy_task_one_ohnecu = (correct_tone_ohne / total_tone) * 100
        error_task_one_ohnecu = 1 - (correct_tone_ohne / total_tone)

        return accuracy_task_one, accuracy_task_two, accuracy_task_one_ohnecu, error_task_one, error_task_two, error_task_one_ohnecu
    

"""
Author: Sina Baghal
Date: August 5, 2024
Description: This project implements the methodologies presented in the paper Deep Smoothing of the Implied Volatility Surface by Ackerer et al (2020). 
Version: 1.0.0
"""

from utils import *
import QuantLib as ql
from scipy.interpolate import interp1d
from scipy.fftpack import fft
from scipy.interpolate import UnivariateSpline
import cvxpy as cv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Function
from scipy.optimize import curve_fit
import numpy as np ; import pandas as pd ; from matplotlib import pyplot as plt
from collections import defaultdict ; from itertools import product 
import os



epsilon = 1e-8

## Bates model parameters -- use for building a reference model
alpha = 0.7; beta = -0.03 ; kappa = 0.5 ; v0 = 0.01 ; theta = 0.0625 ; 
rho = -0.75 ; sigma = 1 ; lam = 0.2

# ### Bates model parameters -- use for building subsequent models
# alpha = 0.5; beta = -0.04 ; kappa = 0.4 ; v0 = 0.02 ; theta = 0.05 ; 
# rho = -0.65 ; sigma = 0.8 ; lam = 0.3

### market data 
spot = 1; rate = 0.0; q = 0.0

## Moneyness #log K/S
ks = [np.log(x) for x in (0.3, 0.4, 0.6, 0.8, 0.9, 0.95, 0.975, 1,
1.025, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2, 2.5, 3)] 
## Strikes
Ks = spot*np.exp(ks)
n_k = len(ks)

## Time to maturity 
Times = ql.TimeGrid(2, 52*12)
taus_0 = \
[Times[idx] for idx in [3,6,6*2,6*3]+[26*x for x in range(1,13)]+[52*6+52*3,52*12]]
n_t = len(taus_0)
today = ql.Date(1, 7, 2020)
# import pdb; pdb.set_trace()
## Auxilary taus 
taus_aux = np.exp(np.linspace(np.log(1/365), np.log(np.max(taus_0)),20))

## Asymptotic behavior
k_asym = [6*np.min(ks),4*np.min(ks),4*np.max(ks),6*np.max(ks)]
K_asym = np.exp(k_asym)
I_large_m = list(product(k_asym,taus_aux))

# Calendar, butterfly arbitrages
k_aux = [k**3 for k in np.linspace(-np.power(-2*np.min(ks),1/3),np.power(2*np.max(ks),1/3),20)] 
K_butcal = np.exp(k_aux)
I_butcal = list(product(k_aux,taus_aux))

I_atm = list(product(np.array([0.0]),taus_aux))
# import pdb;pdb.set_trace()
day_count = ql.Actual365Fixed()
calendar = ql.TARGET()


def cf_bates(tau,u):


    u2, sigma2, alpha2, ju = u**2,sigma**2, alpha**2, 1j*u

    def log_phi_j():
        term1 = -0.5*u2*alpha2+ju*(np.log(1+beta)-0.5*alpha2)
        term2 = np.exp(term1)-1
        return tau*lam*term2

    a = kappa-ju*rho*sigma
    gamma = np.sqrt((sigma2)*(u2+ju)+a**2)
    b = 0.5*gamma*tau
    c = kappa*theta/(sigma2)

    coshb, sinhb = np.cosh(b), np.sinh(b)
    
    term1 = c*tau*a+ju*(tau*(rate-lam*beta)+np.log(spot))
    term2 = coshb+(a/gamma)*sinhb
    term3 = (u2+ju)*v0/(gamma*(coshb/sinhb)+a)
    res = log_phi_j()+term1-term3-2*c*np.log(term2)
    
    return np.exp(res)

N = 2**14  # number of knots for FFT
B = 1000  # integration limit

alpha_corrs = np.array([[[0.4,0.5,0.7,1]]])
n_a = alpha_corrs.shape[2]

du = B / N
u = np.array(np.arange(N)* du).reshape(1,-1,1) 

w_fft = 3 + (-1) ** (np.arange(N) + 1)   # Simpson weights
w_fft[0] = 1 # [1, 4, 2, ..., 4, 2, 4]
dk = 2*np.pi/(du*N) # dk * du = 2pi/N
upper_b = N * dk / 2
lower_b = -upper_b
kus = lower_b + dk * np.arange(N)

taus_0_np = np.array(taus_0).reshape(-1,1,1) #n_t*1*1
w_fft = w_fft.reshape(1,-1,1) # 1*N*1
kus = kus.reshape(1,-1,1) # 1*N*1

cust_interp1d = lambda x,y: interp1d(x, y, kind='linear')
fn_vec = np.vectorize(cust_interp1d, signature='(n),(n)->()')

term1 = np.exp(-rate*taus_0_np)/(alpha_corrs**2+alpha_corrs-u**2+1j*(2*alpha_corrs+1)*u)
term2 = cf_bates(taus_0_np,u-(alpha_corrs+1)*1j) 
mdfd_cf = term1*term2

integrand = np.exp(1j * upper_b * u)*mdfd_cf*w_fft* du / (3*np.pi)
vectorized_fft = np.vectorize(lambda i, j: fft(integrand[i,:, j]), signature='(),()->(n_k)')
tau_indices, corr_indices = np.indices((n_t, n_a))
integrand_fft = vectorized_fft(tau_indices, corr_indices).transpose(0, 2, 1)

Ck_u =np.exp(-alpha_corrs*kus)*np.real(integrand_fft )

f1 = Ck_u.transpose(1, 0, 2).reshape(N, -1)
fn = fn_vec([kus[0,:,0] for _ in range(n_t*n_a)],[f1[:,i] for i in range(n_t*n_a)])

spline = fn.reshape(n_t, n_a) 
vectorized_spline = np.vectorize(lambda i, j, k: spline[i, j](ks[k]), signature='(),(),()->()')

tau_indices, corr_indices, ks_indices  = np.indices((n_t, n_a, n_k))
prices = vectorized_spline(tau_indices, corr_indices, ks_indices)
prices_ = prices.copy()
prices = np.median(prices,axis=1)
print("Bates Model Pricing !") 

@handle_error_with_default(-1)
def quantlib_iv(strike, spot,r0,q0,calculation_date,expiry_date,option_price):

    exercise = ql.EuropeanExercise(expiry_date)

    ql.Settings.instance().evaluationDate = calculation_date

    payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
    option = ql.EuropeanOption(payoff,exercise)
    S = ql.QuoteHandle(ql.SimpleQuote(spot))
    r = ql.YieldTermStructureHandle(ql.FlatForward(0, calendar, r0, day_count))
    q = ql.YieldTermStructureHandle(ql.FlatForward(0, calendar, q0, day_count))
    sigma = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(0, calendar, 0.20, day_count))
    process = ql.BlackScholesMertonProcess(S,q,r,sigma)
    
    iv = option.impliedVolatility(option_price, process)
    return iv

res = defaultdict(lambda: [])
for tau_index in range(n_t):

    tau = taus_0[tau_index]
    expiry_date = today+ql.Period(f"{int(tau*365)}d")
    
    for k_index in range(n_k):
    
        k,K = ks[k_index], Ks[k_index]
        option_price = prices[tau_index,k_index]
        iv = quantlib_iv(K, spot, rate,q,today,expiry_date,option_price)
        res['Price'].append(option_price)
        res['Strike'].append(K)
        res['Moneyness'].append(k)
        res['Tau'].append(tau)
        res['IV'].append(iv)
        if iv>=0:
            res['Variance'].append(iv**2*tau)
        else:
            res['Variance'].append(-1)

df_bates = pd.DataFrame.from_dict(res)
df_bates.to_csv('bates.csv')
df = df_bates[df_bates.IV>=0]
print("Volatility Calculation") # Volatility Calculation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df_torch = torch.tensor(df.values)
df_torch_gpu = df_torch.to(device)

spot_df = df[df.Strike == spot]
spot_taus = np.array(spot_df.Tau)
spot_variances = np.array(spot_df.Variance)
z = cv.Variable(len(spot_taus))
objective = cv.Minimize(cv.sum_squares(z - spot_variances))
constraints = [z[i] >= z[i-1]+1e-10 for i in range(1,len(spot_taus))]
prob = cv.Problem(objective, constraints)
prob.solve(verbose=False)
spot_variances_mdfd= z.value

spline_curve = UnivariateSpline(spot_taus, spot_variances_mdfd, k=5)
spline_curve.set_smoothing_factor(0.5)
# spline_curve_der = spline_curve.derivative()

def poly(x, *coeffs):
    return sum(c * x**i for i, c in enumerate(coeffs))

def poly_derivative(x, *coeffs):
    return sum(i * c * x**(i-1) for i, c in enumerate(coeffs) if i > 0)

degree = 5
popt, _ = curve_fit(poly, taus_aux,  spline_curve(taus_aux), p0=np.ones(degree+1))
popt_tensor = torch.tensor(popt).to(device)
print("ATM Variance Calculation") 

class SPLFunction(Function):
    @staticmethod
    def forward(ctx, tau):
        ctx.save_for_backward(tau)
        return poly(tau,*popt_tensor)

    @staticmethod
    def backward(ctx, grad_output):
        tau, = ctx.saved_tensors
        grad_input = poly_derivative(tau,*popt_tensor)*grad_output
        return grad_input
    
def apply_spline(tau): return SPLFunction.apply(tau)

###############################  Prior Model ###############################

class Wssvi(nn.Module):
    def __init__(self):
        super().__init__()
        self.initialize_weights()

    def initialize_weights(self):
        
        self.gamma = nn.Parameter(torch.tensor(0.5))
        self.eta   = nn.Parameter(torch.tensor(1.0))
        self.rho   = nn.Parameter(torch.tensor(0.0))

    def forward(self, k, tau):

        t1 = apply_spline(tau)
        phi = self.eta*torch.pow(t1,-self.gamma**2)*torch.pow(1+t1,self.gamma**2-1)
        t2 = 1+self.rho*phi*k+torch.sqrt((phi*k+self.rho)**2+1-self.rho**2)
        return t1*t2/2  

###############################  NN Model ###############################

class oneplustanh(nn.Module):
    def __init__(self):
        super().__init__()

        self.alpha_nn = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return self.alpha_nn*(1 +epsilon+ torch.tanh(x))
    
min_learning_rate, learning_rate =  0.01*(0.5)**3, 0.01

class NN(nn.Module):
    

    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()

        layers = []
        in_size = input_size
        for hidden_size in hidden_sizes:
            
            layers.append(nn.Linear(in_size, hidden_size).to(torch.float64))
            layers.append(nn.Softplus())
            in_size = hidden_size
        layers.append(nn.Linear(in_size, output_size).to(torch.float64))
        layers.append(oneplustanh())
        self.layers = nn.ModuleList(layers)
        self.network = nn.Sequential(*layers)

        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                n_in = layer.in_features
                n_out = layer.out_features
                std = (1 / (n_in + n_out))**0.5
                nn.init.normal_(layer.weight, mean=0, std=std)
                nn.init.normal_(layer.bias, mean=0, std=std)
    
    def forward(self, x):
        # return torch.tensor([[1]]).to(device)
        return self.network(x)

def reinitialize_model(model,weight_init,scheduler_class, scheduler_args, lr=learning_rate):

    if weight_init:
        model.initialize_weights()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = scheduler_class(optimizer, **scheduler_args)
    return optimizer, scheduler

relu = nn.ReLU()

def loss_butcal(model_prior, model_nn, ks, ts):
    
    kts = torch.stack((ks, ts), dim=1)
    output = model_prior(ks, ts)*model_nn(kts).squeeze(1)
    
    dw = torch.autograd.grad(outputs=output, inputs=[ks,ts], grad_outputs= torch.ones_like(output), create_graph=True, retain_graph=True)
    dwdk_,dwdtau_ = dw
    d2wdk_ = torch.autograd.grad(outputs=dwdk_, inputs=ks, grad_outputs=torch.ones_like(dwdk_), create_graph=True)[0]

    loss_but = relu(-((1-ks*dwdk_/(2*output))**2-(dwdk_/4)*(1/output+0.25)+d2wdk_/2)).mean()
    loss_cal = relu(-dwdtau_).mean()

    return loss_but, loss_cal 

def loss_large_m(model_prior, model_nn, ks, ts):

    kts = torch.stack((ks,ts), dim=1)
    output = model_prior(ks, ts)*model_nn(kts).squeeze(1)
    dw= torch.autograd.grad(outputs=output, inputs=ks, grad_outputs=torch.ones_like(output), create_graph=True)
    dwdk = dw[0]
    grad_output_dwdk = torch.ones_like(dwdk)
    d2wdk = torch.autograd.grad(outputs=dwdk, inputs=ks, grad_outputs=grad_output_dwdk, create_graph=True)[0]
    loss_large_m = torch.abs(d2wdk).mean()

    return loss_large_m

def loss_atm(model_nn, ks, ts):
    
    kts = torch.stack((ks,ts), dim=1)
    output = model_nn(kts).squeeze(1)
    loss_atm = torch.sqrt(torch.pow(1-output,2).sum()/output.shape[0])
    
    return loss_atm

def loss_0(model_prior, model_nn):

    output = model_prior(df_torch_gpu[:,2], df_torch_gpu[:,3])*(model_nn(df_torch_gpu[:,2:4]).squeeze(1))
    sigma_theta = torch.sqrt(output/df_torch_gpu[:,3])
    temp = sigma_theta-df_torch_gpu[:,4]
    rmse = torch.sqrt(torch.pow(temp,2).mean())
    mape = (torch.abs(temp)/df_torch_gpu[:,4]).mean()

    return rmse, mape 

def total_loss(model_prior, model_nn):

    ###########  Butterfly & Calendar Loss function ###############

    k_but_cal   = torch.tensor(ext(0,I_butcal), requires_grad=True).to(device)
    t_but_cal   = torch.tensor(ext(1,I_butcal), requires_grad=True).to(device) 
    l_but, l_cal = loss_butcal(model_prior, model_nn,k_but_cal, t_but_cal)
    
    ###########  Large moneyness loss function ################

    k_large_m  = torch.tensor(ext(0,I_large_m), requires_grad=True).to(device)
    t_large_m  = torch.tensor(ext(1,I_large_m), requires_grad=True).to(device)
    l_large_m = loss_large_m(model_prior, model_nn, k_large_m, t_large_m)
    
    ###########  ATM loss function ###############

    k_atm = torch.tensor(ext(0,I_atm)).to(device)
    t_atm = torch.tensor(ext(1,I_atm)).to(device)
    l_atm = loss_atm(model_nn, k_atm, t_atm)
    
    ###########  L0 loss function ###############

    rmse, mape  = loss_0(model_prior, model_nn)

    ###########  Total loss ###############

    return rmse, mape, l_cal,l_but,l_large_m,l_atm

def perturb_weights(model, noise_std=0.1):

    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn_like(param) * noise_std
            param.add_(noise)


##################################################

nn_a,nn_b = 40,4 
reg_param = 4

model_prior = Wssvi().to(device)
input_size, hidden_sizes, output_size = 2, [nn_a]*nn_b, 1
model_nn = NN(input_size, hidden_sizes, output_size).to(device)
scheduler_args = {'mode':'min', 'factor':0.5, 'patience':100, 'threshold':0.01, 'min_lr':min_learning_rate}
print(f'Neural network hidden sizes = {nn_a}*{nn_b}')


if os.path.exists('refmodel_nn.pth') and os.path.exists('refmodel_prior.pth'):

    model_nn.load_state_dict(torch.load('refmodel_nn.pth'))
    model_prior.load_state_dict(torch.load('refmodel_prior.pth'))

    model_nn.eval();model_prior.eval()

       
    rmse, mape, l_cal,l_but,l_large_m,l_atm = total_loss(model_prior, model_nn)
    best_loss = rmse+mape+reg_param*(l_cal+l_but+l_large_m)+0.1*l_atm
 
    print('Best loss = ', best_loss)
    print("Loaded reference model !") 
    

else:

    print("Building model from scratch !") 

optimizer_prior = optim.Adam(model_prior.parameters(), lr=learning_rate)
scheduler_prior  =  optim.lr_scheduler.ReduceLROnPlateau(optimizer_prior, **scheduler_args)

optimizer_nn = optim.Adam(model_nn.parameters(), lr=learning_rate)
scheduler_nn = optim.lr_scheduler.ReduceLROnPlateau(optimizer_nn, **scheduler_args)

if os.path.exists('model_nn.pth'):

    os.remove('model_nn.pth')
    os.remove('model_prior.pth')
    print("Temp saved models removed! Save model with different name, if you like to keep them !")



header_message = f"{'Epoch':>5} | {'Best Loss':>12} | {'Curr Loss':>12} | {'RMSE':>12} | {'MAPE':>12} | {'Loss large m':>12} | {'Calendar':>12} | {'Butterfly':>12} | {'Loss ATM':>12} | {'Learning Rate':>13} | {'Tot_Epoch':>5}"
print(header_message)


best_loss = float('inf')
epoch, tot_epoch = 0, 0
model_nn.train();model_prior.train()
loss_df_dict = defaultdict(lambda: [])

while best_loss > 0.02 and tot_epoch < 500000:

    epoch, tot_epoch = epoch+1, tot_epoch+1


    ## there is a check point every 500 epochs
    ck_point_int = 500
    ck_point    = epoch % ck_point_int == 0 

    ## after 4 check points re-init if loss is not less than 1
    is_init_bad = epoch == (4*ck_point_int) and best_loss >1 

    ## every two check points set lr = learning_rate if loss is not small enough
    lr_reboot   =  epoch % (4*ck_point_int)  == 0 and best_loss > 0.05 
    
    if is_init_bad:

        print("Re-init DL model !") 
        print(header_message)
        optimizer_nn, scheduler_nn = reinitialize_model(model_nn, True, optim.lr_scheduler.ReduceLROnPlateau, scheduler_args)
        optimizer_prior, scheduler_prior = reinitialize_model(model_prior, True, optim.lr_scheduler.ReduceLROnPlateau, scheduler_args)
        best_loss = float('inf')
        epoch = 0
        continue  
    
    
    if ck_point:
        
        isfar = loss >= best_loss*1.1 and loss > 0.1

        if isfar and epoch > ck_point_int :

            print(f"Diverging! Restarting from best so far ! Loss value  = {loss} >0.1")
            model_nn.load_state_dict(torch.load('model_nn.pth'))
            model_prior.load_state_dict(torch.load('model_prior.pth'))
            optimizer_nn, scheduler_nn = reinitialize_model(model_nn, False, optim.lr_scheduler.ReduceLROnPlateau, scheduler_args, lr=cur_lr)
            optimizer_prior, scheduler_prior = reinitialize_model(model_prior, False, optim.lr_scheduler.ReduceLROnPlateau, scheduler_args, lr =cur_lr)
            epoch = check_point
            loss = best_loss 
            continue 

        if lr_reboot: 
                                                                                                                                                           
            optimizer_nn, scheduler_nn = reinitialize_model(model_nn, False, optim.lr_scheduler.ReduceLROnPlateau, scheduler_args, lr=learning_rate)
            optimizer_prior, scheduler_prior = reinitialize_model(model_prior, False, optim.lr_scheduler.ReduceLROnPlateau, scheduler_args, lr=learning_rate)
            print("LR restarted !")
            
        perturb_weights(model_nn, noise_std=0.01)
        print("Model purturbed !") 

        check_point = epoch
        print(header_message)
    
    optimizer_prior.zero_grad()
    optimizer_nn.zero_grad()

    rmse, mape, l_cal,l_but,l_large_m,l_atm = total_loss(model_prior, model_nn)
    loss = rmse+mape+reg_param*(l_cal+l_but+l_large_m)+0.1*l_atm

    if torch.isnan(loss): 

        print('Null values detected! Breaking!')
        break 

    if  loss < best_loss:

        best_loss = loss
        torch.save(model_prior.state_dict(), 'model_prior.pth')
        torch.save(model_nn.state_dict(), 'model_nn.pth')
    
    loss.backward()
    optimizer_nn.step()
    optimizer_prior.step()
    
    scheduler_prior.step(loss)
    scheduler_nn.step(loss)

    cur_lr = optimizer_nn.state_dict()["param_groups"][0]["lr"]
    if epoch% 100 == 0 : 

        loss_df_dict = update_loss_df(loss_df_dict, epoch,best_loss,loss,rmse,mape,l_large_m,l_cal,l_but,l_atm,cur_lr,tot_epoch)
        print(f"{epoch:>5} | {best_loss:>12.6f} | {loss:>12.6f} | {rmse:>12.6f} | {mape:>12.6f} | {l_large_m:>12.6f} | {l_cal:>12.6f} | {l_but:>12.6f} | {l_atm:>12.6f} | {cur_lr:>13.10f} | {tot_epoch:>5}")
        
    
print(f'Best loss value = {best_loss}')
plot_ssvi(taus_0,ks,model_prior,model_nn,df_bates[df_bates.IV>=0])
pd.DataFrame.from_dict(loss_df_dict).to_csv('training_loss.csv')
save_model(taus_0,ks,model_nn,model_prior,'model_0.csv')
save_model(taus_aux,k_aux,model_nn,model_prior,'model_aux.csv')
plot3d = lambda k,tau: plot3D(k,tau,model_nn, model_prior) 
plot_surface(plot3d,(0, max(ks)), (min(taus_0), max(taus_0)), num_points=100)


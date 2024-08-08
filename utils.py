import torch
import random
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import cvxpy as cv
import QuantLib as ql

from torch.autograd import Function, gradcheck
from torch.autograd.function import once_differentiable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def handle_error_with_default(default_value):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                return default_value
        return wrapper
    return decorator


ext = lambda i,setofpairs: [pair[i] for pair in setofpairs]

def save_model(taus,ks,model_nn,model_prior,csv_path):
    
    model_nn.eval();model_prior.eval()
    results = []
    for tau in taus:
            for k in ks:
                k, tau = torch.tensor(k).to(device),torch.tensor(tau).to(device)
                ktau = torch.stack((k,tau))
                new_w = model_prior(k,tau)*model_nn(ktau)
                results.append([tau.item(), k.item(),new_w.item()])

    df_model = pd.DataFrame(results, columns=['tau', 'k', 'w'])
    df_model.to_csv(csv_path, index=False)


def update_loss_df(loss_df, epoch,best_loss,loss,rmse,mape,l_large_m,l_cal,l_but,l_atm,cur_lr,tot_epoch):

    loss_df['Epoch'].append(epoch)
    loss_df['BestLoss'].append(best_loss.item())
    loss_df['CurLoss'].append(loss.item())
    loss_df['RMSE'].append(rmse.item())
    loss_df['MAPE'].append(mape.item())
    loss_df['AsymLoss'].append(l_large_m.item())
    loss_df['CalLoss'].append(l_cal.item())
    loss_df['ButLoss'].append(l_but.item())
    loss_df['ATMLoss'].append(l_atm.item())
    loss_df['LR'].append(cur_lr)
    loss_df['TotEpoch'].append(tot_epoch)

    return loss_df


def plot3D(k,tau,model_nn, model_prior):

    model_nn.eval();model_prior.eval()
    k, tau = torch.tensor(k).to(device),torch.tensor(tau).to(device)
    ktau = torch.stack((k,tau),dim=2)
    with torch.no_grad():

        log_iv = torch.log(torch.sqrt(model_prior(k,tau)*model_nn(ktau).squeeze(2)/tau))
    
    return log_iv

def plot_surface(func, k_range, tau_range, num_points=100):
   
   
    ks = torch.linspace(k_range[0], k_range[1], num_points, dtype=torch.float64)
    taus = torch.linspace(tau_range[0], tau_range[1], num_points, dtype=torch.float64)
    
    k_grid, tau_grid = torch.meshgrid(torch.exp(ks), taus, indexing='ij')
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(k_grid.cpu(), tau_grid.cpu(), func(k_grid, tau_grid).cpu(), cmap='viridis')
    
    ax.set_xlabel('Moneyness K/Spot')
    ax.set_ylabel('Time to Maturity Tau')
    ax.set_zlabel('Log Implied Volatility Log(IV)')
    ax.set_title('Log Implied Volatility Surface')
    plt.savefig('Logimplied.png')
    plt.show()

def plot_ssvi(taus,ks,model_prior,model_nn,df_bates):

    fig, ax = plt.subplots(figsize=(10, 6))
    for tau in taus:
        ws = []
        for k in ks:
            k, tau = torch.tensor(k).to(device),torch.tensor(tau).to(device)
            ktau = torch.stack((k,tau))
            new_w = model_prior(k,tau)*model_nn(ktau)
            ws.append(new_w.item())
    
        ax.plot([k.item() for k in ks],ws,marker ='o',linestyle='--', label=f'{np.round(tau.item(),2)}')
    
    for item in df_bates.groupby(by='Tau'): 

        row = item[1]
        tau  = row.Tau.values[0]
        ax.plot(np.log(row.Strike),row.Variance, marker ='+', linestyle='-', label = f'{np.round(tau,2)}')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.savefig('w_theta.png')
    plt.show()


def plot_training():

    df = pd.read_csv('results/synthetic/ref/training_loss.csv')[['TotEpoch',	'BestLoss',	'RMSE',	'MAPE',	'AsymLoss',	'CalLoss',	'ButLoss','ATMLoss']]
    # df = pd.read_csv('results/synthetic/v4/training_loss.csv')[['TotEpoch',	'BestLoss']]
    
    
    df.set_index('TotEpoch', inplace=True)

    # import pdb;pdb.set_trace()
    plt.figure()
    for column in df.columns:
        plt.plot(df.index, df[column], label=column)

    plt.xlabel('Epoch')
    plt.ylabel('Loss values')
    plt.title('Training Metrics Over Epochs')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

if __name__ == "__main__":

    plot_training()

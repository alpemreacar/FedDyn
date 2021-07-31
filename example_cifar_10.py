from utils_general import *
from utils_methods import *

# Dataset initialization

########
# For 'CIFAR100' experiments
#     - Change the dataset argument from CIFAR10 to CIFAR100.
########
# For 'mnist' experiments
#     - Change the dataset argument from CIFAR10 to mnist.
########
# For 'emnist' experiments
#     - Download emnist dataset from (https://www.nist.gov/itl/products-and-services/emnist-dataset) as matlab format and unzip it in "Data/Raw/" folder.
#     - Change the dataset argument from CIFAR10 to emnist.
########
# For Shakespeare experiments
# First generate dataset using LEAF Framework and set storage_path to the data folder
# storage_path = 'LEAF/shakespeare/data/'
#     - In IID use

# name = 'shakepeare'
# data_obj = ShakespeareObjectCrop(storage_path, dataset_prefix)

#     - In non-IID use
# name = 'shakepeare_nonIID'
# data_obj = ShakespeareObjectCrop_noniid(storage_path, dataset_prefix)
#########


# Generate IID or Dirichlet distribution
# IID
n_client = 100
data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, rule='iid', unbalanced_sgm=0)

###
model_name         = 'cifar10' # Model type
com_amount         = 1000
save_period        = 200
weight_decay       = 1e-3
batch_size         = 50
act_prob           = 1
lr_decay_per_round = 1
epoch              = 5
learning_rate      = 0.1
print_per          = 5

# Model function
model_func = lambda : client_model(model_name)
init_model = model_func()
# Initalise the model for all methods or load it from a saved initial model
init_model = model_func()
if not os.path.exists('Output/%s/%s_init_mdl.pt' %(data_obj.name, model_name)):
    print("New directory!")
    os.mkdir('Output/%s/' %(data_obj.name))
    torch.save(init_model.state_dict(), 'Output/%s/%s_init_mdl.pt' %(data_obj.name, model_name))
else:
    # Load model
    init_model.load_state_dict(torch.load('Output/%s/%s_init_mdl.pt' %(data_obj.name, model_name)))    
    
# Methods    
####
print('FedDyn')

alpha_coef = 1e-2
[fed_mdls_sel_FedFyn, trn_perf_sel_FedFyn, tst_perf_sel_FedFyn,
 fed_mdls_all_FedFyn, trn_perf_all_FedFyn, tst_perf_all_FedFyn,
 fed_mdls_cld_FedFyn] = train_FedDyn(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,
                                     epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
                                     model_func=model_func, init_model=init_model, alpha_coef=alpha_coef,
                                     save_period=save_period, lr_decay_per_round=lr_decay_per_round)

###
print('SCAFFOLD')
n_data_per_client = np.concatenate(data_obj.clnt_x, axis=0).shape[0] / n_client
n_iter_per_epoch  = np.ceil(n_data_per_client/batch_size)
n_minibatch = (epoch*n_iter_per_epoch).astype(np.int64)
print_per_ = print_per*n_iter_per_epoch

[fed_mdls_sel_SCAFFOLD, trn_perf_sel_SCAFFOLD, tst_perf_sel_SCAFFOLD,
 fed_mdls_all_SCAFFOLD, trn_perf_all_SCAFFOLD,
 tst_perf_all_SCAFFOLD] = train_SCAFFOLD(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate,
                                         batch_size=batch_size, n_minibatch=n_minibatch, com_amount=com_amount,
                                         print_per=print_per_, weight_decay=weight_decay, model_func=model_func,
                                         init_model=init_model, save_period=save_period, lr_decay_per_round=lr_decay_per_round)
    
####
print('FedAvg')

[fed_mdls_sel_FedAvg, trn_perf_sel_FedAvg, tst_perf_sel_FedAvg,
 fed_mdls_all_FedAvg, trn_perf_all_FedAvg,
 tst_perf_all_FedAvg] = train_FedAvg(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,
                                     epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
                                     model_func=model_func, init_model=init_model, save_period=save_period,
                                     lr_decay_per_round=lr_decay_per_round)
        
#### 
print('FedProx')

mu = 1e-4

[fed_mdls_sel_FedProx, trn_perf_sel_FedProx, tst_perf_sel_FedProx,
 fed_mdls_all_FedProx, trn_perf_all_FedProx,
 tst_perf_all_FedProx] = train_FedProx(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,
                                     epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
                                     model_func=model_func, init_model=init_model, save_period=save_period,
                                     mu=mu, lr_decay_per_round=lr_decay_per_round)

# Plot results
plt.figure(figsize=(6, 5))
plt.plot(np.arange(com_amount)+1, tst_perf_all_FedFyn[:,1], label='FedDyn')
plt.ylabel('Test Accuracy', fontsize=16)
plt.xlabel('Communication Rounds', fontsize=16)
plt.legend(fontsize=16, loc='lower right', bbox_to_anchor=(1.015, -0.02))
plt.grid()
plt.xlim([0, com_amount+1])
plt.title(data_obj.name, fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('Output/%s/plot.pdf' %data_obj.name, dpi=1000, bbox_inches='tight')
# plt.show() 
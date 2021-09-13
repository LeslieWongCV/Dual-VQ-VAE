import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import utils
from models.vqvae import VQVAE
from torchvision.utils import save_image

parser = argparse.ArgumentParser()

"""
Hyperparameters
"""
timestamp = utils.readable_timestamp()

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_updates", type=int, default=50000)
parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--n_residual_hiddens", type=int, default=32)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--n_embeddings", type=int, default=512)
parser.add_argument("--beta", type=float, default=.25)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--dataset", type=str, default='CIFAR10')
parser.add_argument("--method", type=str, default=1)  # method 1, 2, 3, 4
parser.add_argument("--load", type=str, default=False)

# whether or not to save model
parser.add_argument("-save", action="store_true", default=True)
parser.add_argument("--filename", type=str, default=timestamp)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.save:
    print('Results will be saved in ./results/vqvae_' + args.filename + '.pth')

"""
Load data and define batch data loaders
"""

training_data, validation_data, training_loader, validation_loader, x_train_var = utils.load_data_and_data_loaders(
    args.dataset, args.batch_size)
"""
Set up VQ-VAE model with components defined in ./models/ folder
"""
if args.method == 2:
    from models.vqvae_2_cat import VQVAE
elif args.method == 3:
    from models.vqvae_3_add import VQVAE
elif args.method == 4:
    from models.vqvae_4_cat import VQVAE

model = VQVAE(args.n_hiddens, args.n_residual_hiddens,
              args.n_residual_layers, args.n_embeddings, args.embedding_dim, args.beta).to(device)
"""
Load saved model 
"""

if args.load:
    pthfile = 'PATH TO .pth'
    model.load_state_dict(torch.load(pthfile)['model'])

"""
Set up optimizer and training loop
"""
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

model.train()

results = {
    'n_updates': 0,
    'recon_errors': [],
    'loss_vals': [],
    'perplexities': [],

    'n_updates_valid': 0,
    'recon_errors_valid': [],
    'loss_vals_valid': [],
    'perplexities_valid': [],
}


def train():
    for i in range(args.n_updates):
        (x, _) = next(iter(training_loader))
        x = x.to(device)
        optimizer.zero_grad()

        embedding_loss, x_hat, perplexity, kld_loss, MI_loss = model(x)
        recon_loss = torch.mean((x_hat - x) ** 2) / x_train_var
        loss = recon_loss + embedding_loss + kld_loss + MI_loss  # loss = recon_loss for orin VQ-VAE

        loss.backward(loss.clone().detach())
        optimizer.step()

        results["recon_errors"].append(recon_loss.cpu().detach().numpy())
        results["perplexities"].append(perplexity.cpu().detach().numpy())
        results["loss_vals"].append(loss.cpu().detach().numpy())
        results["n_updates"] = i

        if i % args.log_interval == 0:

            (x_valid, _) = next(iter(validation_loader))
            x_valid = x_valid.to(device)
            embedding_loss_valid, x_hat_valid, perplexity_valid, kld_loss_valid, MI_loss_valid = model(x_valid)
            recon_loss_valid = torch.mean((x_hat_valid - x_valid) ** 2) / x_train_var
            loss_valid = recon_loss_valid + embedding_loss_valid + kld_loss_valid + MI_loss_valid
            results["recon_errors_valid"].append(recon_loss_valid.cpu().detach().numpy())
            results["perplexities_valid"].append(perplexity_valid.cpu().detach().numpy())
            results["loss_vals_valid"].append(loss_valid.cpu().detach().numpy())
            results["n_updates_valid"] = i
            """
            save model and print values
            """
            if args.save:
                hyperparameters = args.__dict__
                utils.save_model_and_results(
                    model, results, hyperparameters, args.filename)

            print('Update #', i, 'Recon Error:',
                  np.mean(results["recon_errors"][-args.log_interval:]),
                  'Loss', np.mean(results["loss_vals"][-args.log_interval:]),
                  'Perplexity:', np.mean(results["perplexities"][-args.log_interval:]),

                  '| Valid_Recon Error:',
                  np.mean(recon_loss_valid.cpu().detach().numpy()),
                  'Valid_Loss', np.mean(loss_valid.cpu().detach().numpy()),
                  'Valid_Perplexity:', np.mean(perplexity_valid.cpu().detach().numpy()),
                  )


def test():
    n = 8
    (x, _) = next(iter(training_loader))
    x = x.to(device)
    embedding_loss, x_hat, perplexity, kld_loss, MI_loss = model(x)
    comparison = torch.cat([x[:n], x_hat.view(args.batch_size, -1, 32, 32)[:n]])
    save_image(comparison.cpu(), 'results/' + str(args.method) + '_reconstruction_{}.png'.format(args.filename, nrow=n))


if __name__ == "__main__":
    train()
    test()

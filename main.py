import argparse
from generator import Generator
from critic import Critic
from dataloaders import *
from runner import Runner

parser = argparse.ArgumentParser(description='Pytorch Implementation of WGAN-GP.')
parser.add_argument('--mode', choices=['train', 'generate'], default='train', help='train or generate?')
parser.add_argument('--dataset', choices=['cifar10', 'celeba', 'mnist'], default='cifar10')
parser.add_argument('--img_height', type=int, default=32)
parser.add_argument('--img_width', type=int, default=32)
parser.add_argument('--img_channels', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--z_dim', type=int, default=128)
parser.add_argument('--channels', type=int, default=128, help='number of channels generator and critic convolutions')
parser.add_argument('--g_lr', type=float, default=0.0002)
parser.add_argument('--d_lr', type=float, default=0.0002)
parser.add_argument('--g_optimizer', choices=['adam', 'rmsprop'], default='adam')
parser.add_argument('--d_optimizer', choices=['adam', 'rmsprop'], default='adam')
parser.add_argument('--gp_lambda', type=float, default=10.0)
parser.add_argument('--max_steps', type=int, default=100000, help='total number of generator steps')
parser.add_argument('--num_critic_steps', type=int, default=5, help='critic steps per generator step')
args = parser.parse_args()

device = "cuda" if tc.cuda.is_available() else "cpu"

## Dataloaders.
if args.dataset == 'celeba':
    dataloader, _ = get_celeba_dataloaders(batch_size=args.batch_size)
elif args.dataset == 'cifar10':
    dataloader, _ = get_cifar10_dataloaders(batch_size=args.batch_size)
elif args.dataset == 'mnist':
    dataloader, _ = get_mnist_dataloaders(batch_size=args.batch_size)
else:
    raise NotImplementedError

## Models.
g_model = Generator(
    img_height=args.img_height,
    img_width=args.img_width,
    img_channels=args.img_channels,
    z_dim=args.z_dim,
    channels=args.channels
).to(device)

d_model = Critic(
    img_height=args.img_height,
    img_width=args.img_width,
    img_channels=args.img_channels,
    channels=args.channels
).to(device)

## Optimizers.
if args.g_optimizer == 'adam':
    g_optimizer = tc.optim.Adam(g_model.parameters(), lr=args.g_lr, betas=(0.0, 0.90))
elif args.g_optimizer == 'rmsprop':
    g_optimizer = tc.optim.RMSprop(g_model.parameters(), lr=args.g_lr, alpha=0.90)
else:
    raise NotImplementedError

if args.d_optimizer == 'adam':
    d_optimizer = tc.optim.Adam(d_model.parameters(), lr=args.d_lr, betas=(0.0, 0.90))
elif args.d_optimizer == 'rmsprop':
    d_optimizer = tc.optim.RMSprop(d_model.parameters(), lr=args.d_lr, alpha=0.90)
else:
    raise NotImplementedError

## Schedulers.
g_scheduler = tc.optim.lr_scheduler.OneCycleLR(
    optimizer=g_optimizer,
    max_lr=args.g_lr,
    total_steps=args.max_steps,
    pct_start=0.0,
    anneal_strategy='linear',
    cycle_momentum=False,
    div_factor=1.0)

d_scheduler = tc.optim.lr_scheduler.OneCycleLR(
    optimizer=d_optimizer,
    max_lr=args.d_lr,
    total_steps=args.max_steps,
    pct_start=0.0,
    anneal_strategy='linear',
    cycle_momentum=False,
    div_factor=1.0)

## Runner.
runner = Runner(
    batch_size=args.batch_size,
    max_steps=args.max_steps,
    g_model=g_model,
    d_model=d_model,
    g_optimizer=g_optimizer,
    d_optimizer=d_optimizer,
    g_scheduler=g_scheduler,
    d_scheduler=d_scheduler,
    dataloader=dataloader,
    gp_lambda=args.gp_lambda,
    num_critic_steps=args.num_critic_steps)

if args.mode == 'train':
    runner.train()
elif args.mode == 'generate':
    runner.generate(64)

import torch as tc
from utils import save_img_grid, compute_grad2
import os


class Runner:
    def __init__(self, device, batch_size, max_steps, dataloader, g_model, d_model, g_optimizer, d_optimizer,
                 g_scheduler, d_scheduler, gp_lambda, num_critic_steps, checkpoint_dir, model_dir):

        self.device = device
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.g_model = g_model
        self.d_model = d_model
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.g_scheduler = g_scheduler
        self.d_scheduler = d_scheduler
        self.dataloader = dataloader
        self.gp_lambda = gp_lambda
        self.num_critic_steps = num_critic_steps
        self.checkpoint_dir = checkpoint_dir
        self.model_dir = model_dir

        self.reference_noise = 2.0 * tc.rand(size=(64, self.g_model.z_dim)) - 1.0
        self.global_step = 0

    def train_critic(self, x_real, x_fake):
        ## trains critic one step.
        x_real, x_fake = x_real.to(self.device), x_fake.to(self.device)
        x_real.requires_grad_()
        x_fake.requires_grad_()
        self.d_optimizer.zero_grad()

        d_real = self.d_model(x_real)[:,0]
        d_fake = self.d_model(x_fake)[:,0]
        d_loss_wgan = d_fake.mean() - d_real.mean()
        d_loss_wgan.backward(retain_graph=True)

        interp_s = tc.rand(size=(self.batch_size,))
        interp_s = interp_s.to(self.device)
        interp_s = interp_s.view(-1, 1, 1, 1)
        x_interp = interp_s * x_real + (1. - interp_s) * x_fake
        d_interp = self.d_model(x_interp)[:,0]
        gp = self.gp_lambda * tc.square(tc.sqrt(compute_grad2(d_interp, x_interp)) - 1.0).mean()
        gp.backward()

        self.d_optimizer.step()
        return d_loss_wgan

    def train_generator(self, x_fake):
        ## trains generator one step.
        x_fake = x_fake.to(self.device)
        d_fake = self.d_model(x_fake)[:,0]
        g_loss = -d_fake.mean()

        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        return g_loss

    def train(self):
        epoch = 1
        while self.global_step < self.max_steps:
            print(f"Epoch {epoch}\n-------------------------------")
            for batch_idx, (x_real, _) in enumerate(self.dataloader, 1):
                if x_real.shape[0] != self.batch_size:
                    continue

                # refactored code to train critic of different batches of data at each critic step.
                # there are num_critic_steps per generator step.
                x_fake = self.generate(self.batch_size)
                d_loss = self.train_critic(x_real, x_fake)

                # generator step.
                if batch_idx % self.num_critic_steps == 0:
                    x_fake = self.generate(self.batch_size)
                    g_loss = self.train_generator(x_fake)

                    self.g_scheduler.step()
                    self.d_scheduler.step()

                    self.global_step += 1

                    if True: #self.global_step % 10 == 0:
                        print("[{}/{}] Generator Loss: {}... Critic Loss: {} ".format(
                            self.global_step, self.max_steps, g_loss.item(), d_loss.item()))

                    if self.global_step % 10 == 0:
                        self.generate_and_save(None, z=self.reference_noise)
                        self.save_checkpoint()

            epoch += 1
        return

    def generate(self, num_samples, z=None):
        if z is None:
            z = 2.0 * tc.rand(size=(num_samples, self.g_model.z_dim)) - 1.0  # uniform noise in [-1, 1]^z_dim
        z = z.to(self.device)
        x_fake = self.g_model(z)
        return x_fake

    def generate_and_save(self, num_samples, z=None):
        samples = self.generate(num_samples, z=z).detach()
        samples = samples.cpu()
        fp = save_img_grid(images=samples, grid_size=8)
        print('Saved images to {}'.format(fp))

    def save_checkpoint(self):
        model_path = os.path.join(self.checkpoint_dir, self.model_dir)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        g_model_path = os.path.join(model_path, 'g_model.pth')
        d_model_path = os.path.join(model_path, 'd_model.pth')
        g_optimizer_path = os.path.join(model_path, 'g_opt.pth')
        d_optimizer_path = os.path.join(model_path, 'd_opt.pth')

        tc.save(self.g_model.state_dict(), g_model_path)
        tc.save(self.d_model.state_dict(), d_model_path)
        tc.save(self.g_optimizer.state_dict(), g_optimizer_path)
        tc.save(self.d_optimizer.state_dict(), d_optimizer_path)

        print('Successfully saved checkpoints to {}'.format(model_path))

    def maybe_load_checkpoint(self):
        model_path = os.path.join(self.checkpoint_dir, self.model_dir)
        g_model_path = os.path.join(model_path, 'g_model.pth')
        d_model_path = os.path.join(model_path, 'd_model.pth')
        g_optimizer_path = os.path.join(model_path, 'g_opt.pth')
        d_optimizer_path = os.path.join(model_path, 'd_opt.pth')

        try:
            self.g_model.load_state_dict(tc.load(g_model_path))
            self.d_model.load_state_dict(tc.load(d_model_path))
            self.g_optimizer.load_state_dict(tc.load(g_optimizer_path))
            self.d_optimizer.load_state_dict(tc.load(d_optimizer_path))
            print('Successfully loaded checkpoints from {}'.format(model_path))
        except Exception:
            print('Bad checkpoint or none. Continuing training from scratch.')


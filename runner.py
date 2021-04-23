import torch as tc
from utils import save_img_grid, compute_grad2

class Runner:
    def __init__(self, batch_size, max_steps, dataloader, g_model, d_model, g_optimizer, d_optimizer,
                 g_scheduler, d_scheduler, gp_lambda, num_critic_steps):

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

        self.reference_noise = 2.0 * tc.rand(size=(64, self.g_model.z_dim)) - 1.0
        self.global_step = 0

    def train_critic(self, x_real, x_fake):
        ## trains critic one step.
        self.d_optimizer.zero_grad()
        x_real.requires_grad_()
        x_fake.requires_grad_()

        d_real = self.d_model(x_real)[:,0]
        d_fake = self.d_model(x_fake)[:,0]
        d_loss_wgan = d_fake.mean() - d_real.mean()
        d_loss_wgan.backward(retain_graph=True)
        #d_loss_wgan.backward()

        interp_s = tc.rand(size=(self.batch_size,)).view(-1, 1, 1, 1)
        x_interp = interp_s * x_real + (1. - interp_s) * x_fake
        d_interp = self.d_model(x_interp)[:,0]
        gp = self.gp_lambda * tc.square(tc.sqrt(compute_grad2(d_interp, x_interp)) - 1.0).mean()
        gp.backward()

        self.d_optimizer.step()
        return d_loss_wgan

    def train_generator(self, x_fake):
        ## trains generator one step.
        d_fake = self.d_model(x_fake)[:,0]
        g_loss = -d_fake.mean()

        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        return g_loss

    def train(self):
        while self.global_step < self.max_steps:
            for batch_idx, (x_real, _) in enumerate(self.dataloader, 1):

                for _ in range(self.num_critic_steps):
                    x_fake = self.generate(self.batch_size)
                    d_loss = self.train_critic(x_real, x_fake)

                x_fake = self.generate(self.batch_size)
                g_loss = self.train_generator(x_fake)

                self.g_scheduler.step()
                self.d_scheduler.step()

                self.global_step += 1

                if True: #self.global_step % 10 == 0:
                    print("[{}/{}] Generator Loss: {}... Critic Loss: {} ".format(
                        self.global_step, self.max_steps, g_loss.item(), d_loss.item()))

                if self.global_step % 50 == 0:
                    self.generate_and_save(None, z=self.reference_noise)

        return

    def generate(self, num_samples, z=None):
        if z is None:
            z = 2.0 * tc.rand(size=(num_samples, self.g_model.z_dim)) - 1.0  # uniform noise in [-1, 1]^z_dim
        x_fake = self.g_model(z)
        return x_fake

    def generate_and_save(self, num_samples, z=None):
        samples = self.generate(num_samples, z=z).detach()
        fp = save_img_grid(images=samples, grid_size=8)
        print('Saved images to {}'.format(fp))



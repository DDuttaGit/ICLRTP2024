from model import *
import scipy.stats as stats


generator = Generator()
discriminator = Discriminator()


if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-3)
optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=1e-3)

def train(E, noisetype, mu, sigma):
    data_loader = DataModule(dim=(100,50,1), f='gauss', mu=float(mu), sigma=float(sigma))
    Xt_train, Xt_test = data_loader.X_train, data_loader.X_test


    epochs = int(E)
    clip_value = 0.2
    n_critic = 7


    batches_done = 0
    for e in range(epochs):
        for b, f in enumerate(data_loader.X_train):
            dimn = (f.size(dim=0),  f.size(dim=1))

            if noisetype == "U":
                z = Variable(Tensor(np.random.random(size=dimn)))
            else:
                z = Variable(Tensor(np.random.random(size=dimn)))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()


            # Generate a batch of images
            f_tilde = generator(z).detach()
            # Adversarial loss
            d_loss = -torch.mean(discriminator(f)) + torch.mean(discriminator(f_tilde))

            d_loss.backward()
            optimizer_D.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-clip_value, clip_value)

            # Train the generator every n_critic iterations
            if e % n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Generate a batch of images
                f_tilde = generator(z)
                # Adversarial loss
                g_loss = -torch.mean(discriminator(f_tilde))

                g_loss.backward()
                optimizer_G.step()

            batches_done += 1
            sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G Loss: %f] [Generated: %f]"%
                             (e+1, epochs, b+1, len(Xt_train), d_loss.item(), g_loss.item(), torch.mean(f_tilde)))


    # Test the Generated Values
    z = Variable(Tensor(np.random.random(size=(1000,1))))
    f_tilde = (generator(z)).flatten() * data_loader.scale
    x1 = f_tilde.cpu().detach().numpy()
    x2 = Xt_test.flatten().cpu().detach().numpy() * data_loader.scale
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    x1.sort()
    x2.sort()
    ax1.scatter(x1, x2, label='Direct QQ-plot on Tar vs Gen', color='blue')

    hist = torch.histc(torch.tensor(x1), bins=10, min=x1.min(), max=x1.max())
    bins = 10
    x = range(bins)
    ax2.bar(x, hist, align='center')


    plt.show()

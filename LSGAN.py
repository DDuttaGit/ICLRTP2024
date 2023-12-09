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
    data_loader = DataModule(dim=(100,50,1), f='gauss', mu=float(mu), sigma=float(sigma), normalize=False)
    Xt_train, Xt_test = data_loader.X_train, data_loader.X_test

    epochs = int(E)
    for e in range(epochs):
        for b, f in enumerate(data_loader.X_train):
            dimn = (f.size(dim=0),  f.size(dim=1))

            lbl_one = Variable(Tensor(np.ones(dimn)), requires_grad=False)
            if noisetype == "U":
                z = Variable(Tensor(np.random.random(size=dimn)))
            else:
                z = Variable(Tensor(np.random.random(size=dimn)))
            f_tilde = generator(z)

            optimizer_D.zero_grad()
            lbl_zero = Variable(Tensor(np.zeros(dimn)), requires_grad=False)
            errD_real = 0.5 * torch.mean((discriminator(f) - lbl_one)**2)
            errD_fake = 0.5 * torch.mean((discriminator(f_tilde) - lbl_zero)**2)
            d_loss = (errD_fake + errD_real) / 2
            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            g_loss = 0.5 * torch.mean((discriminator(f_tilde.detach()) - lbl_one)**2)
            g_loss.backward()
            optimizer_G.step()


        sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G Loss: %f] [Generated: %f]"%
                             (e+1, epochs, b+1, len(Xt_train), d_loss.item(), g_loss.item(), torch.mean(f_tilde)))
                             
    # Test the Generated Values
    z = Variable(Tensor(np.random.random(size=(1000,1))))
    f_tilde = (generator(z)).flatten() # * data_loader.scale
    x1 = f_tilde.cpu().detach().numpy()
    x2 = Xt_test.flatten().cpu().detach().numpy() # * data_loader.scale
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    x1.sort()
    x2.sort()
    ax1.scatter(x1, x2, label='Direct QQ-plot on Tar vs Gen', color='blue')

    hist = torch.histc(torch.tensor(x1), bins=10, min=x1.min(), max=x1.max())
    bins = 10
    x = range(bins)
    ax2.bar(x, hist, align='center')

    plt.show()

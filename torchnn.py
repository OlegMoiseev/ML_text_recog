# Import dependencies
import torch
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Grayscale, Resize
from torchvision.transforms.functional import pil_to_tensor


# Get data 
train = datasets.MNIST(root="data_num_recog", download=False, train=True, transform=ToTensor())
dataset = DataLoader(train, 32)


# 1,28,28 - classes 0-9

# Image Classifier Neural Network
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (28 - 6) * (28 - 6), 10)
        )

    def forward(self, x):
        return self.model(x)


# Instance of the neural network, loss, optimizer
clf = ImageClassifier().to('cuda')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training flow 
if __name__ == "__main__":
    # for epoch in range(10):  # train for 10 epochs
    #     for batch in dataset:
    #         X, y = batch
    #         X, y = X.to('cuda'), y.to('cuda')
    #         yhat = clf(X)
    #         loss = loss_fn(yhat, y)
    #
    #         # Apply backprop
    #         opt.zero_grad()
    #         loss.backward()
    #         opt.step()
    #
    #     print(f"Epoch:{epoch} loss is {loss.item()}")
    #     if loss.item() < 1e-5:
    #         break
    #
    # with open('model_state_cuda.pt', 'wb') as f:
    #     save(clf.state_dict(), f)

    with open('model_state.pt', 'rb') as f:
        clf.load_state_dict(load(f))

    # img = Image.open('data_num_recog/img_6.jpg')
    for i in range(16):
        img = Image.open('data_num_recog/roi/roi{0}.jpg'.format(i))
        # img = Grayscale(1)(img)
        img = Resize((28, 28))(img)
        # img.show()
        # img_gray = Grayscale(1)(img)
        # img_tensor = pil_to_tensor(img_gray).unsqueeze(0)
        img_tensor = ToTensor()(img).unsqueeze(0).to('cuda')
        print("roi" + str(i) + '.jpg: ' + str(torch.argmax(clf(img_tensor))))

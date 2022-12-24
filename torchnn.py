# Import dependencies
import torch
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Grayscale, Resize
from collections import Counter
import tessract
import cv2


# 1,28,28 - classes 0-9

# Image Classifier Neural Network
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # self.model = nn.Sequential(
        #     nn.Conv2d(1, 32, (3, 3)),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, (3, 3)),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, (3, 3)),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(64 * (28 - 6) * (28 - 6), 10)
        # )

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * (28 - 8) * (28 - 8), 10)
        )

    def forward(self, x):
        return self.model(x)


def recog_nums(input_img):
    """
    :param input_img: numpy array with image
    :return: counted numbers from image, Counter
    """
    clf = ImageClassifier().to('cuda')

    with open('model_state_cuda_4_1.pt', 'rb') as f:
        clf.load_state_dict(load(f))

    src_gray = cv2.blur(input_img, (3, 3))

    thresh = 255  # initial threshold
    rois = tessract.thresh_callback(thresh, src_gray)
    list_nums = []
    for roi in rois:
        roi = Image.fromarray(roi)
        img = Grayscale(1)(roi)
        img = Resize((28, 28))(img)

        img_tensor = ToTensor()(img).unsqueeze(0).to('cuda')
        torched = torch.argmax(clf(img_tensor)).item()
        list_nums.append(torched)

    counted_nums = Counter(list_nums)

    return counted_nums


# Training flow 
if __name__ == "__main__":
    # Get data
    # train = datasets.MNIST(root="data_num_recog", download=True, train=True, transform=ToTensor())
    # dataset = DataLoader(train, 50)
    #
    # Instance of the neural network, loss, optimizer
    clf = ImageClassifier().to('cuda')
    opt = Adam(clf.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    #
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
    #     # if loss.item() < 1e-5:
    #     #     break
    #
    # with open('model_state_cuda_4_1.pt', 'wb') as f:
    #     save(clf.state_dict(), f)

    with open('model_state_cuda_4_1.pt', 'rb') as f:
        clf.load_state_dict(load(f))

    list_nums = []
    for i in range(17):
        img = Image.open('data_num_recog/roi/roi{0}.jpg'.format(i))
        img = Grayscale(1)(img)
        img = Resize((28, 28))(img)
        # img.show()
        # img_tensor = pil_to_tensor(img_gray).unsqueeze(0)
        img_tensor = ToTensor()(img).unsqueeze(0).to('cuda')
        torched = torch.argmax(clf(img_tensor)).item()
        list_nums.append(torched)
        print("roi" + str(i) + '.jpg: ' + str(torched))
    counted_nums = Counter(list_nums)
    print(counted_nums)

{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "4775a057",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "PyTorch version: 2.14.0+cu130\n",
      "GPU: NVIDIA GeForce GTX 1650\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import torchvision\n",
    "import torchvision.transforms as transforms\n",
    "\n",
    "print('PyTorch version:', torch.__version__)\n",
    "if torch.cuda.is_available():\n",
    "    print('GPU:', torch.cuda.get_device_name(0))\n",
    "    device = torch.device('cuda')\n",
    "else:\n",
    "    print('No GPU found, using CPU.')\n",
    "    device = torch.device('cpu')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d695d151",
   "metadata": {},
   "source": [
    "### KMIST dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "51de522d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Dataset KMNIST\n",
      "    Number of datapoints: 60000\n",
      "    Root location: ./data\n",
      "    Split: Train\n",
      "    StandardTransform\n",
      "Transform: Compose(\n",
      "               ToTensor()\n",
      "           )\n",
      "Dataset KMNIST\n",
      "    Number of datapoints: 10000\n",
      "    Root location: ./data\n",
      "    Split: Test\n",
      "    StandardTransform\n",
      "Transform: Compose(\n",
      "               ToTensor()\n",
      "           )\n"
     ]
    }
   ],
   "source": [
    "transform = transforms.Compose([transforms.ToTensor()])\n",
    "\n",
    "trainset = torchvision.datasets.KMNIST(\n",
    "    root='./data', train=True, download=True, transform=transform)\n",
    "testset = torchvision.datasets.KMNIST(\n",
    "    root='./data', train=False, download=True, transform=transform)\n",
    "\n",
    "print(trainset)\n",
    "print(testset)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "d81bf57d",
   "metadata": {},
   "outputs": [],
   "source": [
    "BATCH_SIZE = 100 # 60000 datapoints so 600 batches to train\n",
    "train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True)\n",
    "test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9461548f",
   "metadata": {},
   "source": [
    "##### Using VAE to get Latent Space Representation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "070d7450",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "VAE(\n",
      "  (fc1): Linear(in_features=784, out_features=512, bias=True)\n",
      "  (fc2): Linear(in_features=512, out_features=256, bias=True)\n",
      "  (fc_mu): Linear(in_features=256, out_features=2, bias=True)\n",
      "  (fc_log_var): Linear(in_features=256, out_features=2, bias=True)\n",
      "  (fc3): Linear(in_features=2, out_features=256, bias=True)\n",
      "  (fc4): Linear(in_features=256, out_features=512, bias=True)\n",
      "  (fc5): Linear(in_features=512, out_features=784, bias=True)\n",
      ")\n"
     ]
    }
   ],
   "source": [
    "class VAE(nn.Module):\n",
    "    def __init__(self, x, hl1, hl2, z):  # x->input, hl1 & hl2 hidden layers, z->latent space\n",
    "        super().__init__()\n",
    "\n",
    "        # Encoder\n",
    "        self.fc1 = nn.Linear(x, hl1)\n",
    "        self.fc2 = nn.Linear(hl1, hl2)\n",
    "        self.fc_mu = nn.Linear(hl2, z)\n",
    "        self.fc_log_var = nn.Linear(hl2, z)\n",
    "        # Decoder\n",
    "        self.fc3 = nn.Linear(z, hl2)\n",
    "        self.fc4 = nn.Linear(hl2, hl1)\n",
    "        self.fc5 = nn.Linear(hl1, x)\n",
    "\n",
    "    def encoder(self, x_t):\n",
    "        y = F.relu(self.fc1(x_t))\n",
    "        y = F.relu(self.fc2(y))\n",
    "        return self.fc_mu(y), self.fc_log_var(y)  # mu, log_var\n",
    "\n",
    "    def sampling(self, mu, log_var):\n",
    "        std = torch.exp(0.5 * log_var)\n",
    "        eps = torch.randn_like(std)\n",
    "        return eps.mul(std).add_(mu)\n",
    "\n",
    "    def decoder(self, z_t):\n",
    "        y = F.relu(self.fc3(z_t))\n",
    "        y = F.relu(self.fc4(y))\n",
    "        return torch.sigmoid(self.fc5(y))\n",
    "\n",
    "    def forward(self, x_t):\n",
    "        mu, log_var = self.encoder(x_t.view(-1, 784))\n",
    "        z_t = self.sampling(mu, log_var)\n",
    "        return self.decoder(z_t), mu, log_var\n",
    "\n",
    "vae = VAE(x=784, hl1=512, hl2=256, z=2).to(device)\n",
    "if torch.cuda.is_available():\n",
    "    vae.cuda()\n",
    "    \n",
    "print(vae)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "be5f32b2",
   "metadata": {},
   "outputs": [],
   "source": [
    "def loss_function(recon_x, x, mu, log_var):\n",
    "    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')\n",
    "    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())\n",
    "    return BCE + KLD"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "62bd5877",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1: loss 258.8405 | pixel acc 0.8467\n",
      "Epoch 2: loss 258.0985 | pixel acc 0.8474\n",
      "Epoch 3: loss 257.5988 | pixel acc 0.8479\n",
      "Epoch 4: loss 256.7622 | pixel acc 0.8486\n",
      "Epoch 5: loss 256.0964 | pixel acc 0.8493\n",
      "Epoch 6: loss 255.5653 | pixel acc 0.8498\n",
      "Epoch 7: loss 254.9353 | pixel acc 0.8503\n",
      "Epoch 8: loss 254.3953 | pixel acc 0.8508\n",
      "Epoch 9: loss 253.7635 | pixel acc 0.8514\n",
      "Epoch 10: loss 253.5071 | pixel acc 0.8517\n",
      "Epoch 11: loss 253.0101 | pixel acc 0.8522\n",
      "Epoch 12: loss 252.7775 | pixel acc 0.8524\n",
      "Epoch 13: loss 252.1122 | pixel acc 0.8530\n",
      "Epoch 14: loss 252.0124 | pixel acc 0.8531\n",
      "Epoch 15: loss 251.6093 | pixel acc 0.8536\n",
      "Epoch 16: loss 251.2074 | pixel acc 0.8538\n",
      "Epoch 17: loss 250.9149 | pixel acc 0.8540\n",
      "Epoch 18: loss 250.5388 | pixel acc 0.8544\n",
      "Epoch 19: loss 250.1138 | pixel acc 0.8548\n",
      "Epoch 20: loss 249.9233 | pixel acc 0.8550\n",
      "Epoch 21: loss 249.6115 | pixel acc 0.8552\n",
      "Epoch 22: loss 249.4927 | pixel acc 0.8555\n",
      "Epoch 23: loss 249.2015 | pixel acc 0.8557\n",
      "Epoch 24: loss 248.9568 | pixel acc 0.8559\n",
      "Epoch 25: loss 248.6403 | pixel acc 0.8561\n",
      "Epoch 26: loss 248.4598 | pixel acc 0.8564\n",
      "Epoch 27: loss 248.1262 | pixel acc 0.8567\n",
      "Epoch 28: loss 247.8389 | pixel acc 0.8569\n",
      "Epoch 29: loss 248.7528 | pixel acc 0.8563\n",
      "Epoch 30: loss 248.1648 | pixel acc 0.8567\n",
      "Epoch 31: loss 247.4868 | pixel acc 0.8572\n",
      "Epoch 32: loss 247.3790 | pixel acc 0.8573\n",
      "Epoch 33: loss 246.9568 | pixel acc 0.8576\n",
      "Epoch 34: loss 247.6382 | pixel acc 0.8572\n",
      "Epoch 35: loss 246.7392 | pixel acc 0.8578\n",
      "Epoch 36: loss 246.5732 | pixel acc 0.8580\n",
      "Epoch 37: loss 246.3602 | pixel acc 0.8581\n",
      "Epoch 38: loss 246.2618 | pixel acc 0.8584\n",
      "Epoch 39: loss 246.1927 | pixel acc 0.8584\n",
      "Epoch 40: loss 246.1186 | pixel acc 0.8584\n",
      "Epoch 41: loss 246.5028 | pixel acc 0.8582\n",
      "Epoch 42: loss 246.3259 | pixel acc 0.8583\n",
      "Epoch 43: loss 245.6288 | pixel acc 0.8588\n",
      "Epoch 44: loss 245.1960 | pixel acc 0.8591\n",
      "Epoch 45: loss 245.4219 | pixel acc 0.8591\n",
      "Epoch 46: loss 245.1900 | pixel acc 0.8592\n",
      "Epoch 47: loss 244.9250 | pixel acc 0.8594\n",
      "Epoch 48: loss 245.0480 | pixel acc 0.8594\n",
      "Epoch 49: loss 244.7931 | pixel acc 0.8596\n",
      "Epoch 50: loss 244.7320 | pixel acc 0.8597\n"
     ]
    }
   ],
   "source": [
    "optimizer  = torch.optim.AdamW(vae.parameters())\n",
    "EPOCHS = 50\n",
    "\n",
    "train_loss = []\n",
    "train_accs = []\n",
    "for epoch in range(EPOCHS):\n",
    "    vae.train()\n",
    "    running_loss, running_correct, n_train = 0.0, 0, 0\n",
    "    for images, _ in train_loader:\n",
    "        images = images.to(device)\n",
    "        optimizer.zero_grad()\n",
    "\n",
    "        recon, mu, log_var = vae(images)\n",
    "        loss = loss_function(recon, images, mu, log_var)\n",
    "\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "\n",
    "        batch_size = images.size(0)\n",
    "        running_loss += loss.item()\n",
    "        n_train      += batch_size\n",
    "        pred   = recon > 0.5\n",
    "        target = images.view(-1, 784) > 0.5\n",
    "        running_correct += (pred == target).sum().item()\n",
    "\n",
    "    train_loss.append(running_loss / n_train)\n",
    "    train_accs.append(running_correct / (n_train * 784))\n",
    "    print(f\"Epoch {epoch+1}: loss {train_loss[-1]:.4f} | pixel acc {train_accs[-1]:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "aaa5f1d4",
   "metadata": {},
   "outputs": [],
   "source": [
    "for epoch in range(EPOCHS):\n",
    "    vae.eval()\n",
    "    val_loss = 0\n",
    "    with torch.no_grad():\n",
    "        for image, _ in test_loader:\n",
    "            image = image.cuda()\n",
    "            recon, mu, log_var = vae(image)\n",
    "\n",
    "            val_loss += loss_function(recon,image,mu,log_var).item()\n",
    "    val_loss = val_loss/len(test_loader)\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "4dbf19a2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Files Saved\n"
     ]
    }
   ],
   "source": [
    "def extract(dataset):\n",
    "    with torch.no_grad():\n",
    "        vae.eval()\n",
    "        loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False)\n",
    "\n",
    "        latent_space = []\n",
    "        for x, _ in loader:\n",
    "            x = x.to(device)\n",
    "            x = x.view(-1,784)\n",
    "            mu, log_var = vae.encoder(x) \n",
    "            latent_space.append(mu.cpu())\n",
    "        return torch.cat(latent_space)\n",
    "\n",
    "Z_train = extract(trainset)\n",
    "Z_test = extract(testset)\n",
    "np.save(\"Z_train.npy\", Z_train.numpy())\n",
    "np.save(\"Z_test.npy\",  Z_test.numpy())\n",
    "print(\"Files Saved\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": ".venv (3.14.8)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.14.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

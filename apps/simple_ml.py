"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl

import needle.nn as nn
from apps.models import *
import time
device = ndl.cpu()

def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    # changed: starter names this image_filesname
    with gzip.open(image_filesname, "rb") as img_file:
      magic, num_image, rows, cols = struct.unpack(">IIII", img_file.read(16))
      image_data = np.frombuffer(img_file.read(), dtype=np.uint8)
      X = image_data.reshape(num_image, rows * cols).astype(np.float32)
      X = X / 255.0
    
    with gzip.open(label_filename, "rb") as lbl_file:
      magic, num_labels = struct.unpack(">II", lbl_file.read(8))
      y = np.frombuffer(lbl_file.read(), dtype=np.uint8).astype(np.int8)
    
    return X, y
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    batch_size = Z.shape[0]
    log_sum_exp = ndl.log(ndl.summation(ndl.exp(Z), axes=(1,)))
    z_y = ndl.summation(Z * y_one_hot, axes=(1,))
    return ndl.summation(log_sum_exp - z_y) / batch_size
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    num_examples = X.shape[0]
    num_classes = W2.shape[1]

    for i in range(0, num_examples, batch):
      X_batch = X[i : i + batch]
      y_batch = y[i : i + batch]

      X_tensor = ndl.Tensor(X_batch)

      y_one_hot = np.zeros((y_batch.shape[0], num_classes))
      y_one_hot[np.arange(y_batch.shape[0]), y_batch] = 1
      y_tensor = ndl.Tensor(y_one_hot)

      logits = ndl.relu(X_tensor @ W1) @ W2
      loss = softmax_loss(logits, y_tensor)
      loss.backward()

      W1 = ndl.Tensor(W1.numpy() - lr * W1.grad.numpy())
      W2 = ndl.Tensor(W2.numpy() - lr * W2.grad.numpy())

    return W1, W2
    ### END YOUR SOLUTION

### CIFAR-10 training ###
def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    import gc

    if opt is None:
        model.eval()
    else:
        model.train()

    model_device = model.parameters()[0].device

    total_correct = 0
    total_loss = 0.0
    total_samples = 0

    for X, y in dataloader:
        X = ndl.Tensor(X, device=model_device, requires_grad=False)
        y = ndl.Tensor(y, device=model_device, requires_grad=False)

        logits = model(X)
        loss = loss_fn(logits, y)

        batch_size = X.shape[0]

        total_loss += loss.numpy().item() * batch_size
        total_samples += batch_size

        total_correct += np.sum(
            np.argmax(logits.numpy(), axis=1) == y.numpy()
        )

        if opt is not None:
            opt.reset_grad()
            loss.backward()
            opt.step()

        gc.collect()

    return (
        total_correct / total_samples,
        total_loss / total_samples,
    )
    ### END YOUR SOLUTION


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    for epoch in range(n_epochs):
        start = time.time()
        avg_acc, avg_loss = epoch_general_cifar10(
            dataloader,
            model,
            loss_fn=loss_fn(),
            opt=opt,
        )
        print(f"epoch {epoch + 1}/{n_epochs}: acc {avg_acc:.4f}, loss {avg_loss:.4f}, {time.time() - start:.0f}s", flush=True)

    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    return epoch_general_cifar10(
        dataloader,
        model,
        loss_fn=loss_fn(),
    )
    ### END YOUR SOLUTION


### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    import gc

    if opt is None:
        model.eval()
    else:
        model.train()

    nbatch, batch_size = data.shape
    h = None

    total_correct = 0
    total_loss = 0.0
    total_samples = 0

    for i in range(0, nbatch - 1, seq_len):
        X, y = ndl.data.get_batch(data, i, seq_len, device=device, dtype=dtype)

        logits, h = model(X, h)
        loss = loss_fn(logits, y)

        if isinstance(h, tuple):
            h = tuple(x.detach() for x in h)
        else:
            h = h.detach()

        num_targets = y.shape[0]

        total_loss += loss.numpy().item() * num_targets
        total_samples += num_targets

        total_correct += np.sum(
            np.argmax(logits.numpy(), axis=1) == y.numpy()
        )

        if opt is not None:
            opt.reset_grad()
            loss.backward()
            if clip is not None:
                opt.clip_grad_norm(clip)
            opt.step()

        gc.collect()

    return (
        total_correct / total_samples,
        total_loss / total_samples,
    )
    ### END YOUR SOLUTION


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    for epoch in range(n_epochs):
        start = time.time()
        avg_acc, avg_loss = epoch_general_ptb(
            data,
            model,
            seq_len=seq_len,
            loss_fn=loss_fn(),
            opt=opt,
            clip=clip,
            device=device,
            dtype=dtype,
        )
        print(f"epoch {epoch + 1}/{n_epochs}: acc {avg_acc:.4f}, loss {avg_loss:.4f}, {time.time() - start:.0f}s", flush=True)

    return avg_acc, avg_loss
    ### END YOUR SOLUTION

def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    return epoch_general_ptb(
        data,
        model,
        seq_len=seq_len,
        loss_fn=loss_fn(),
        device=device,
        dtype=dtype,
    )
    ### END YOUR SOLUTION

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)

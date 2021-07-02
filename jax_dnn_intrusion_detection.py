import jax.numpy as jnp
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, Dropout, Sigmoid
from jax.experimental import optimizers
from jax import jit, grad
from jax import random
import numpy.random as npr
import itertools
from sklearn.preprocessing import Normalizer
from datasets.intrusion_detection.read_csv import get_training_data, get_testing_data


def get_net_arch(mode='train'):
    net_init, net_apply = stax.serial(
        Dense(512), Relu,
        Dropout(0.95, mode=mode),
        Dense(64), Relu,
        Dropout(0.95, mode=mode),
        Dense(32), Relu,
        Dense(16), Relu,
        Dense(8), Relu,
        Dense(1),
        Sigmoid
    )
    return net_init, net_apply


def train_net(test_folder=None, num_epochs=3, step_size=1e-4):
    input_dim = 41
    batch_size = 64

    # Use stax to set up network initialization and evaluation functions
    key = random.PRNGKey(0)
    net_init, net_apply = get_net_arch(mode='train')
    _, net_apply_test = get_net_arch(mode='test')

    # Binary Cross-Entropy loss (logistic regression)
    def cross_entropy(predictions, targets):
        preds = predictions.squeeze()
        label_probs = preds * targets + (1 - preds) * (1 - targets)
        return -jnp.mean(jnp.log(label_probs))

    def loss_cross_entropy(params, batch, key_):
        inputs_, targets = batch
        predictions = net_apply(params, inputs_, rng=key_)
        return cross_entropy(predictions, targets)

    def accuracy_precision_recall(params, batch, key_):
        inputs_, targets = batch
        predictions = net_apply_test(params, inputs_, rng=key_)
        pred = jnp.round(predictions.squeeze())
        accuracy = jnp.mean(pred == targets)

        true_positive = jnp.sum(pred * targets)
        true_negative = jnp.sum(pred == targets) - true_positive
        false_positive = jnp.sum(pred) - true_positive
        false_negative = targets.shape[0] - true_positive - true_negative - false_positive

        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)

        return accuracy, precision, recall

    # Define a compiled update step
    @jit
    def step(iteration, opt_state_, batch, key_):
        params = get_params(opt_state_)
        g = grad(loss_cross_entropy)(params, batch, key_)
        return opt_update(iteration, g, opt_state_)

    train_data = get_training_data('datasets/intrusion_detection/')
    test_data = get_testing_data('datasets/intrusion_detection/')
    train_x = train_data.iloc[:, 1:42].values
    train_y = train_data.iloc[:, 0].values
    test_x = test_data.iloc[:, 1:42].values
    test_y = test_data.iloc[:, 0].values
    num_train = train_x.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    scaler = Normalizer().fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    def data_stream():
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                yield train_x[batch_idx], train_y[batch_idx]

    batches = data_stream()

    # Initialize parameters, not committing to a batch shape
    in_shape = (-1, input_dim)
    out_shape, net_params = net_init(key, in_shape)

    # Use optimizers to set optimizer initialization and update functions
    opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)
    iter_count = itertools.count()

    # Optimize parameters in a loop
    opt_state = opt_init(net_params)

    best_test_acc = 0.0

    for i in range(num_epochs):
        for j in range(num_batches):
            key, subkey = random.split(key)
            opt_state = step(next(iter_count), opt_state, next(batches), subkey)
            if j % 100 == 0:
                net_params = get_params(opt_state)
                loss_train = loss_cross_entropy(net_params, (train_x, train_y), subkey)
                acc_test, precision_test, recall_test = accuracy_precision_recall(net_params, (test_x, test_y), subkey)
                print("Training set loss {}".format(loss_train))
                print("Test set accuracy {}, precision {}, recall {}".format(acc_test, precision_test, recall_test))
                if acc_test > best_test_acc:
                    print("Saved mode with test accuracy", acc_test)
                    best_test_acc = acc_test
                    save_net(test_folder, net_params)

        net_params = get_params(opt_state)
        loss_train = loss_cross_entropy(net_params, (train_x, train_y), subkey)
        acc_test, precision_test, recall_test = accuracy_precision_recall(net_params, (test_x, test_y), subkey)
        print("Epoch", i)
        print("Training set loss {}".format(loss_train))
        print("Test set accuracy {}, precision {}, recall {}".format(acc_test, precision_test, recall_test))

    net_params = get_params(opt_state)
    print(net_params)
    return net_params, net_apply_test


def load_net(test_folder):
    net_params = jnp.load(test_folder + "/net_params_intrusion.npy", allow_pickle=True)
    _, net_apply_test = get_net_arch(mode='test')
    return net_params, net_apply_test


def save_net(test_folder, net_params):
    if test_folder is not None:
        jnp.save(test_folder + "/net_params_intrusion", net_params)


if __name__ == "__main__":
    network_params, net_apply_fun = train_net(test_folder="./", num_epochs=3, step_size=1e-6)
    load_net(test_folder="./")

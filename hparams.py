import tensorflow as tf

def create_hparams(hparams_string=None, verbose=True):
    # Create model hyperparameters. Parse non-default from given string.

    hparams = tf.contrib.training.HParams(
        ## hyperparameters
        NUM_epochs=100,
        snapshot_interval=10,
        eval_interval=1,

        base_lr=1e-3,
        batch_size=64,
        hidden_dim=128,

        ## synthetic dataset
        syn_width = 256,
        syn_height = 32,
        syn_num_training = 10000,
        syn_num_test = 32,

        ## preprocessing
        resize_width = 256,
        resize_height = 32,
        gauss_mean = 50.0,
        gauss_std = 25.0,
        normalize_mean = [0.485, 0.456, 0.406],
        normalize_std = [0.229, 0.224, 0.225]

    )

    if hparams_string:
        tf.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams
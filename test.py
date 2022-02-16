"""Open the dataset, evaluate the model and confront results with expectations."""
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

from utilities.plots import jet_scatter, jet_scatter3D, jet_hist, save_figure
from utilities.file_opener import get_jet_list, get_models, stand_data
from utilities.generator import jet_gen

with tf.device('/GPU:0'):
    jet_list, target = get_jet_list()

    njets = jet_list.shape[0]

    autoencoder_model = get_models()

    jet_tag = np.argmax(target, axis=1)

    jet_list = stand_data(jet_list)

    encoded_features = autoencoder_model.encoder_predict(jet_list, target)
    decoded_jets = autoencoder_model.decoder_predict(encoded_features, target)

    print(decoded_jets[:10, :])
    print(jet_list[:10, :])

    plt.figure()
    plt.subplot(121)
    jet_scatter(encoded_features, jet_tag, 0, 1)
    plt.subplot(122)
    jet_scatter(encoded_features, jet_tag, 0, 2)
    save_figure("encoded_features_2d")

    plt.figure()
    jet_scatter3D(encoded_features, jet_tag)
    save_figure("encoded_features_3d")

    for i in range(5):
        particle_type = np.zeros(5)
        particle_type[i] = 1
        filter_type = np.all((target == particle_type), axis=1)
        encoded_features_filtered = encoded_features[filter_type, :]
        jet_tag_filtered = jet_tag[filter_type]
        plt.figure()
        jet_scatter3D(encoded_features_filtered, jet_tag_filtered)
        save_figure(f"encoded_features_3d_type{i}")

    particle_type = [0, 0, 0, 1, 0]
    particle_tag = np.argmax(particle_type)

    filter_type = np.all((target == particle_type), axis=1)
    jet_list = jet_list[filter_type, :]
    target = target[filter_type, :]

    n_events = np.size(jet_list, 0)

    generated_jets = jet_gen(particle_type, n_events, 789)

    print(generated_jets[:10])

    print(jet_list[:10])
    print(target[:10])

    feature_names = ['pt', 'eta', 'mass', 'tau32_b1', 'tau32_b2']
    jet_hist(jet_list, feature_names,
             custom_name=f'_type{particle_tag}_true', bins=100)
    jet_hist(generated_jets, feature_names,
             custom_name=f'_type{particle_tag}_gen', bins=100)

    plt.show()

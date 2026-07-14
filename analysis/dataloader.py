import pickle

def load_data(fig):
    with open(f'data/data_dict_fig{fig}.pkl', 'rb') as file:
        data_dict = pickle.load(file)
    return data_dict

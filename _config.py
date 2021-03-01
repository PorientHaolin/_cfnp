# ------------
# Dataset args
# ------------

datasets = {}  # name to n_features
# binary class datasets
datasets['a9a'] = {'n_features': 123, 'is_multi': False, 'has_test': True}
datasets['ijcnn1'] = {'n_features': 22, 'is_multi': False, 'has_test': True}
datasets['madelon'] = {'n_features': 500, 'is_multi': False, 'has_test': True}
datasets['mushrooms'] = {'n_features': 112, 'is_multi': False, 'has_test': False}
datasets['phishing'] = {'n_features': 68, 'is_multi': False, 'has_test': False}
datasets['splice'] = {'n_features': 60, 'is_multi': False, 'has_test': False}
datasets['w8a'] = {'n_features': 300, 'is_multi': False, 'has_test': True}

# multi-class dataset
datasets['dna.scale'] = {'n_features': 180, 'is_multi': True, 'has_test': True}
datasets['mnist'] = {'n_features': 780, 'is_multi': True, 'has_test': True}
datasets['pendigits'] = {'n_features': 16, 'is_multi': True, 'has_test': True}
datasets['Sensorless'] = {'n_features': 48, 'is_multi': True, 'has_test': False}
datasets['usps'] = {'n_features': 256, 'is_multi': True, 'has_test': True}
# Nomarlize to binary
n_classes = 2
labels = [0, 1]

import linear_classifier
import oneHidden_reluAct_net
import multilayerperceptron


#                   #
#       models      #
#                   #
def create_model(model_name, dataset, nHiddenLayers=0):
    switcher={
        'linear':linear_classifier.LinearNetwork(dataset.full_data.shape[1], dataset.label_encoder.classes_,
                                name= model_name,
                                mean=dataset.scaler.mean_, variance=dataset.scaler.var_,
                                isInputAbsolute=not dataset.is_relative),
        'simple_nn':oneHidden_reluAct_net.OneHiddenNetwork(dataset.full_data.shape[1], dataset.label_encoder.classes_,
                                name= model_name,
                                mean=dataset.scaler.mean_, variance=dataset.scaler.var_,
                                isInputAbsolute=not dataset.is_relative),
        'multi_layer_perceptron':multilayerperceptron.MultiLayerPerceptron(dataset.full_data.shape[1], dataset.label_encoder.classes_,
                                name= model_name,
                                mean=dataset.scaler.mean_, variance=dataset.scaler.var_,
                                isInputAbsolute=not dataset.is_relative,
                               nHiddenLayers=nHiddenLayers)
    }

    return switcher.get(model_name)
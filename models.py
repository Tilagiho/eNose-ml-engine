import linear_classifier
import oneHidden_reluAct_net


#                   #
#       models      #
#                   #
def create_model(model_name, dataset):
    switcher={
        'linear':linear_classifier.LinearNetwork(dataset.full_data.shape[1], dataset.label_encoder.classes_,
                                name= model_name,
                                mean=dataset.scaler.mean_, variance=dataset.scaler.var_,
                                isInputAbsolute=not dataset.is_relative),
        'simple_nn':oneHidden_reluAct_net.OneHiddenNetwork(dataset.full_data.shape[1], dataset.label_encoder.classes_,
                                name= model_name,
                                mean=dataset.scaler.mean_, variance=dataset.scaler.var_,
                                isInputAbsolute=not dataset.is_relative,
                                nHidden=hidden_layer_width)
    }

    return switcher.get(model_name)
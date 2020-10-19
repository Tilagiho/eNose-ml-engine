import multilayerperceptron


#                   #
#       models      #
#                   #
def create_model(model_name, dataset, nHiddenLayers=0, loss_func=None, input_function="average", output_function="logsoftmax", is_multi_label=False, threshold=0.3):
    switcher={
        'multi_layer_perceptron':multilayerperceptron.MultiLayerPerceptron(dataset.full_data.shape[1], dataset.label_encoder.classes_,
                                name= model_name,
                                mean=dataset.scaler.mean_, variance=dataset.scaler.var_,
                                isInputAbsolute=not dataset.is_relative,
                                nHiddenLayers=nHiddenLayers,
                                loss_func=loss_func)
    }

    return switcher.get(model_name)
def print_layers(model):
    """
    Print the layers of a model in a readable format and some additional information.

    Args:
        model: The model whose layers are to be printed.
    """
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable params: ", trainable_params, "     ||      total params: ", total_params)
    print("percent trainable: ", 100 * trainable_params / total_params, '%')
    # at which layer are trainable params?
    if len(set([(sum(p.numel() for p in layer.parameters() if p.requires_grad), sum(p.numel() for p in layer.parameters())) for layer in model.model.layers])) == 1:
        print("All layers have ", sum(p.numel() for p in model.model.layers[0].parameters() if p.requires_grad), " trainable params out of ", sum(p.numel() for p in model.model.layers[0].parameters()), " total params")
    else:
        for i,layer in enumerate(model.model.layers):
            num_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
            if num_params > 0:
                print("layer ", i, " has ", num_params, " trainable params out of ", sum(p.numel() for p in layer.parameters()), " total params")
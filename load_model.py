from detect_normal import select_device, attempt_load  

def load_model(device_, weights):
    # Initialize
    device = select_device(device_)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location = device)  # load FP32 model
    
    return device, model, half

def combine_model(device_):    
    device, model_plate, half = load_model(device_, weights = ['final_model/plate_model.pt'])
    _, model_char, _ = load_model(device_, weights = ['final_model/char_model.pt'])
    
    return device, half, model_plate, model_char

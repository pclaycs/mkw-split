from pygrabber.dshow_graph import FilterGraph

def find_obs_camera():
    devices = FilterGraph().get_input_devices()
    print("Available cameras:")
    for i, name in enumerate(devices):
        print(f"  {i}: {name}")
    
    for i, name in enumerate(devices):
        if "OBS" in name:
            print(f"\nFound OBS camera at index {i}: {name}")
            return i
    return None
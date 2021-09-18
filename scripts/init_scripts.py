def init_script():
    import sys
    import os
    import socket
    import warnings
    warnings.filterwarnings("ignore")
    # makes the tvg package visible no matter where the scripts
    # are launched from    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(script_dir))

    # only for my local machine - gabT - TOREMOVE
    hostname = socket.gethostname()
    if hostname == "ga1i13o": 
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
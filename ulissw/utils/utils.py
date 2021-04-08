def read_apikey(path):
    with open(path, "r") as f:
    	line=f.readline()
    
    	return line.strip()
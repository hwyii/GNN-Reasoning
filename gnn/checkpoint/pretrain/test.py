import torch  


file_path = "CWQ-final.ckpt"  


checkpoint = torch.load(file_path, weights_only=True)  


print("Checkpoint Keys:", checkpoint.keys())  

 
if "model_state_dict" in checkpoint:  
    print("Model State Dict Keys:")  
    for key, value in checkpoint["model_state_dict"].items():  
        print(f"{key}: {value.shape}")
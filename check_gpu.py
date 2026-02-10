import torch
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(i)
        print('device', i, prop.name, 'total_memory_GB', prop.total_memory/1024**3)
else:
    print('no gpu')

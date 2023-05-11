

base_MACs = 7685502246
base_param = 20884814.0
mode = "train"

with open("/lab/tmpig8e/u/yuecheng/yuecheng_code/ShELL-classification/Xception_src/resnet.txt") as f:
    lines = f.readlines()
    for line in lines:
        analysis = line.strip().split(": ")
        for stuff in analysis:
            if "Conv2d" in stuff and "eConv2d" not in stuff:
                info = stuff[7:-1].split(", ")
                in_channel = int(info[4])
                out_channel = int(info[5])
                kernel_size = int(info[7][-2])
                macs = float(info[2].split()[0])
                conv2d_macs = 0


                if mode == "psp_eval":
                    conv2d_macs = out_channel * in_channel*kernel_size*kernel_size
                    base_param += in_channel*kernel_size*kernel_size
                elif mode == "train":
                    conv2d_macs = 2*macs
                elif mode == "psp_train":
                    conv2d_macs = 2*macs + out_channel * in_channel*kernel_size*kernel_size

                base_MACs += conv2d_macs
            elif "Linear" in stuff:
                info = stuff[7:-1].split(", ")
                in_channel = int(info[4].split("=")[1])
                out_channel = int(info[5].split("=")[1])
                macs = float(info[2].split()[0])
                linear_macs = 0

                if mode == "psp_eval":
                    linear_macs = in_channel
                    base_param += in_channel*kernel_size*kernel_size
                elif mode == "train":
                    linear_macs = 2*macs
                elif mode == "psp_train":
                    linear_macs = 2*macs + in_channel

                base_MACs += linear_macs
                
            elif "AdaptiveAvgPool2d" in stuff:
                info = stuff[len("AdaptiveAvgPool2d")+1:-1].split(", ")
                macs = float(info[2].split()[0])

                if "train" in mode:
                    base_MACs += macs

            elif "BatchNorm2d" in stuff:
                info = stuff[len("BatchNorm2d")+1:-1].split(", ")
                macs = float(info[2].split()[0])

                if "train" in mode:
                    base_MACs += 2*macs
                
            elif "ReLU" in stuff:
                info = stuff[len("ReLU")+1:-1].split(", ")
                macs = float(info[2].split()[0])

                if "train" in mode:
                    base_MACs += macs

            elif "MaxPool2d" in stuff:
                info = stuff[len("MaxPool2d")+1:-1].split(", ")
                macs = float(info[2].split()[0])

                if "train" in mode:
                    base_MACs += macs
    if "train" in mode:
        base_MACs += base_param


            
print(base_MACs)
# import torchvision.models as models
# import torch
# from ptflops import get_model_complexity_info

# with torch.cuda.device(0):
#     import timm
#     model = models.resnet50()
#     model.fc = torch.nn.Linear(2048, 30)
#     macs, params = get_model_complexity_info(model, (3, 299, 299), as_strings=True,
#                                             print_per_layer_stat=True, flops_units="Mac", param_units="", verbose=True)
#     # print(macs, params)
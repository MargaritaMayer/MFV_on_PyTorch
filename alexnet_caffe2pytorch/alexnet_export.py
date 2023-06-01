import caffe
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import numpy as np

def get_weights_from_caffe(prototxt_filename, caffemodel_filename):
  net = caffe.Net(prototxt_filename, caffemodel_filename, caffe.TEST)
  weights = [] 
  for i in range(len(net.layers)): 
    w = [net.layers[i].blobs[bi].data[...] for bi in range(len(net.layers[i].blobs))]
    if len(w) > 0:
	    weights.append(w[0]) 
	    weights.append(w[1]) 
  return weights

def copy_weights_to_pytorch():
	weights = get_weights_from_caffe("deploy_alexnet_updated.prototxt", "bvlc_reference_caffenet.caffemodel")
	for i in range(len(weigts_name)):
	  alexnet_weights[weigts_name[i]] = torch.tensor(weights[i])
	pytorch_net.load_state_dict(alexnet_weights)

def show_classes(probabilities, text):
	with open("imagenet_classes.txt", "r") as f:
	    categories = [s.strip() for s in f.readlines()]
	top5_prob, top5_catid = torch.topk(probabilities, 5)
	print(text)
	for i in range(top5_prob.size(0)):
	    print(categories[top5_catid[i]], top5_prob[i].item())

def show_caffe_result():
	end = 'prob'
	src = caffe_net.blobs['data']
	w = caffe_net.blobs['data'].width
	h = caffe_net.blobs['data'].height
	src.reshape(1,3,h,w)
	src.data[0] = input_tensor.numpy()
	acts = caffe_net.forward(end=end)
	probabilities = torch.tensor(acts[end][0])
	show_classes(probabilities, "Caffe probabilities for img 'dog.jpg':")

def show_pytorch_result():
	pytorch_net.eval()
	with torch.no_grad():
	  r = pytorch_net(input_tensor.unsqueeze(0))
	probabilities = torch.nn.functional.softmax(r[0], dim=0)
	show_classes(probabilities, "PyTorch probabilities for img 'dog.jpg':")


caffe_net = caffe.Classifier("deploy_alexnet_updated.prototxt", "bvlc_reference_caffenet.caffemodel",
                             mean = np.float32([104.0, 117.0, 123.0])) 

class AlexNet(nn.Module):
    def __init__(self, num_classes = 1000, dropout = 0.5):
    	super(AlexNet, self).__init__()
        self.features = nn.Sequential(    
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5),
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

pytorch_net = AlexNet()
alexnet_weights = pytorch_net.state_dict()
weigts_name = list(alexnet_weights.keys())
copy_weights_to_pytorch()



preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(227),
    transforms.ToTensor(),
])
input_image = Image.open("dog.jpg")
input_tensor = 255*preprocess(input_image)-caffe_net.transformer.mean['data']

# Compare neural networks
show_caffe_result()
show_pytorch_result()


# save weights
torch.save(pytorch_net.state_dict(), "net.pt")
# -*- coding:utf-8 _*-
# @License: MIT Licence
# @Author: Forence
# @Contact: wang00sheng@gmail.com
# @GitHub: https://github.com/Forence1999
# @Time: 3/2/2023
import torch
import torch.nn as nn
import torchvision
import numpy as np
import os


def copy_model_params(src_model, tgt_model):
	"""Copy model parameters from src_model to tgt_model."""

	tgt_model.load_state_dict(src_model.state_dict(), strict=True)

	return tgt_model


def get_torch_info():
	print(torch.__version__)
	print(torch.version.cuda)  # cuda版本查询
	print(torch.backends.cudnn.version())  # cudnn版本查询
	print(torch.cuda.get_device_name(0))  # 设备名


def set_random_seed(seed=0):
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def set_device():
	assert False, 'no checking yet'
	os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print("device:", device)
	return device


def clear_gpu_cache():
	torch.cuda.empty_cache()


def name_tensor(tensor):
	assert False, 'no checking yet'

	tensor = torch.rand(3, 4, 1, 2, names=('C', 'N', 'H', 'W'))
	# 使用align_to可以对维度方便地排序
	tensor = tensor.align_to('N', 'C', 'H', 'W')
	tensor.sum('C')
	tensor.select('C', index=0)


def set_default_tensor_type():
	torch.set_default_tensor_type(torch.FloatTensor)


def convert_np_tensor(tensor):
	assert False, 'no checking yet'

	ndarray = tensor.cpu().numpy()
	tensor = torch.from_numpy(ndarray).float()
	tensor = torch.from_numpy(ndarray.copy()).float()  # If ndarray has negative stride.


def permute_tensor(tensor):
	assert False, 'no checking yet'
	tensor = tensor[torch.randperm(tensor.size(0))]


def flip_tensor(tensor):
	assert False, 'no checking yet'
	tensor = tensor[:, :, :, torch.arange(tensor.size(3) - 1, -1, -1).long()]
	tensor = torch.flip(tensor, dims=[3])


def one_hot_tensor(tensor):
	assert False, 'no checking yet'
	tensor = torch.tensor([0, 2, 1, 3])
	N = tensor.size(0)
	num_classes = 4
	one_hot = torch.zeros(N, num_classes).long()
	one_hot.scatter_(dim=1, index=torch.unsqueeze(tensor, dim=1), src=torch.ones(N, num_classes).long())


def tensor_mul(tensor1, tensor2):
	assert False, 'no checking yet'

	# Matrix multiplcation: (m*n) * (n*p) * -> (m*p).
	result = torch.mm(tensor1, tensor2)

	# Batch matrix multiplication: (b*m*n) * (b*n*p) -> (b*m*p)
	result = torch.bmm(tensor1, tensor2)

	# Element-wise multiplication.
	result = tensor1 * tensor2


def model_modules(model):
	model.modules()
	model.children()


def tensorboard():
	assert False, 'no checking yet'

	from torch.utils.tensorboard import SummaryWriter

	writer = SummaryWriter()

	for n_iter in range(100):
		writer.add_scalar('Loss/train', np.random.random(), n_iter)
		writer.add_scalar('Loss/test', np.random.random(), n_iter)
		writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
		writer.add_scalar('Accuracy/test', np.random.random(), n_iter)


def init_model(model, tensor):
	assert False, 'no checking yet'

	for layer in model.modules():
		if isinstance(layer, torch.nn.Conv2d):
			torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out',
										  nonlinearity='relu')
			if layer.bias is not None:
				torch.nn.init.constant_(layer.bias, val=0.0)
		elif isinstance(layer, torch.nn.BatchNorm2d):
			torch.nn.init.constant_(layer.weight, val=1.0)
			torch.nn.init.constant_(layer.bias, val=0.0)
		elif isinstance(layer, torch.nn.Linear):
			torch.nn.init.xavier_normal_(layer.weight)
			if layer.bias is not None:
				torch.nn.init.constant_(layer.bias, val=0.0)

	# Initialization with given tensor.
	layer.weight = torch.nn.Parameter(tensor)


def copy_model_params(src_model, tgt_model):
	return tgt_model.load_state_dict(src_model.state_dict(), strict=True)


def save_load_model(model, path, optimizer):
	assert False, 'no checking yet'

	start_epoch = 0
	# Load checkpoint.
	if resume:  # resume为参数，第一次训练时设为0，中断再训练时设为1
		model_path = os.path.join('model', 'best_checkpoint.pth.tar')
		assert os.path.isfile(model_path)
		checkpoint = torch.load(model_path)
		best_acc = checkpoint['best_acc']
		start_epoch = checkpoint['epoch']
		model.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		print('Load checkpoint at epoch {}.'.format(start_epoch))
		print('Best accuracy so far {}.'.format(best_acc))


def train_model(model, train_loader, optimizer, start_epoch, current_acc, num_epochs, best_acc, resume, ):
	assert False, 'no checking yet'

	import shutil

	# Train the model
	for epoch in range(start_epoch, num_epochs):
		...

		# Test the model
		...

		# save checkpoint
		is_best = current_acc > best_acc
		best_acc = max(current_acc, best_acc)
		checkpoint = {
			'best_acc' : best_acc,
			'epoch'    : epoch + 1,
			'model'    : model.state_dict(),
			'optimizer': optimizer.state_dict(),
		}
		model_path = os.path.join('model', 'checkpoint.pth.tar')
		best_model_path = os.path.join('model', 'best_checkpoint.pth.tar')
		torch.save(checkpoint, model_path)
		if is_best:
			shutil.copy(model_path, best_model_path)


def extract_conv_feature(model, image):
	assert False, 'no checking yet'

	import collections

	# VGG-16 relu5-3 feature.
	model = torchvision.models.vgg16(pretrained=True).features[:-1]
	# VGG-16 pool5 feature.
	model = torchvision.models.vgg16(pretrained=True).features
	# VGG-16 fc7 feature.
	model = torchvision.models.vgg16(pretrained=True)
	model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-3])
	# ResNet GAP feature.
	model = torchvision.models.resnet18(pretrained=True)
	model = torch.nn.Sequential(collections.OrderedDict(
		list(model.named_children())[:-1]))

	with torch.no_grad():
		model.eval()
		conv_representation = model(image)


class FeatureExtractor(torch.nn.Module):
	"""Helper class to extract several convolution features from the given
	pre-trained model.

	Attributes:
		_model, torch.nn.Module.
		_layers_to_extract, list<str> or set<str>

	Example:
		>>> model = torchvision.models.resnet152(pretrained=True)
		>>> model = torch.nn.Sequential(collections.OrderedDict(
				list(model.named_children())[:-1]))
		>>> conv_representation = FeatureExtractor(
				pretrained_model=model,
				layers_to_extract={'layer1', 'layer2', 'layer3', 'layer4'})(image)
	"""

	def __init__(self, pretrained_model, layers_to_extract):
		torch.nn.Module.__init__(self)
		self._model = pretrained_model
		self._model.eval()
		self._layers_to_extract = set(layers_to_extract)

	def forward(self, x):
		with torch.no_grad():
			conv_representation = []
			for name, layer in self._model.named_children():
				x = layer(x)
				if name in self._layers_to_extract:
					conv_representation.append(x)
			return conv_representation


def finetune_MLP():
	# 微调全连接层
	model = torchvision.models.resnet18(pretrained=True)
	for param in model.parameters():
		param.requires_grad = False
	model.fc = nn.Linear(512, 100)  # Replace the last fc layer
	optimizer = torch.optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)

	# 以较大学习率微调全连接层，较小学习率微调卷积层
	model = torchvision.models.resnet18(pretrained=True)
	finetuned_parameters = list(map(id, model.fc.parameters()))
	conv_parameters = (p for p in model.parameters() if id(p) not in finetuned_parameters)
	parameters = [{
		'params': conv_parameters,
		'lr'    : 1e-3
	},
		{
			'params': model.fc.parameters()
		}]
	optimizer = torch.optim.SGD(parameters, lr=1e-2, momentum=0.9, weight_decay=1e-4)


class MyLoss(torch.autograd.Function):
	def forward(self, x):
		return x.view(-1).sum(0)

	def backward(self, x):
		import pdb

		pdb.set_trace()
		return x

# v = torch.autograd.Variable(torch.randn(5, 5), requires_grad=True)
# loss = MyLoss()(v)
# loss.backward()

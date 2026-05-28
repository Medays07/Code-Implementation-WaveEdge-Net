import os
import csv
import Modelloss
import torch
from Multimodal_baseline_model import *
from torch import optim
from evaluation import *
import torch.nn.functional as F
import torchvision
import time


class Solver(object):
	def __init__(self, config, train_loader, valid_loader, test_loader):

		# Data loader
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.test_loader = test_loader

		# Models
		self.unet = None
		self.optimizer = None
		self.scheduler = None
		self.img_ch = config.img_ch
		self.output_ch = config.output_ch

		# Losses
		self.criterion = Loss.DiceLoss()

		# Hyper-parameters
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2

		# Training settings
		self.num_epochs = config.num_epochs
		self.num_epochs_test = config.num_epochs_test
		self.batch_size = config.batch_size

		# Path
		self.model_path = config.model_path
		self.train_result_path = config.train_result_path
		self.val_result_path = config.val_result_path
		self.test_result_path = config.test_result_path
		self.mode = config.mode

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model_type = config.model_type
		self.build_model()

	def build_model(self):
		"""Build generator and discriminator."""
		if self.model_type =='U_Net':
			self.unet = U_Net(img_ch=1,output_ch=1)
		elif self.model_type == 'MSU_Net':
			self.unet = MSU_Net(img_ch=1, output_ch=1)
		elif self.model_type == 'my_Net':
			self.unet = my_Net(img_ch=1, output_ch=1)


		self.optimizer = optim.SGD(self.unet.parameters(), self.lr, momentum=0.9, weight_decay=0.000001)


		self.unet.to(self.device)
		self.print_gpu_memory('After model to GPU')

	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(model)
		print(name)
		print("The number of parameters: {}".format(num_params))

	def reset_grad(self):
		# Zero the gradient buffers.
		self.unet.zero_grad()

	def print_gpu_memory(self, tag=''):
		if not torch.cuda.is_available():
			print('[GPU Memory] CUDA is not available.')
			return

		if self.device.type == 'cuda':
			device_id = self.device.index
			if device_id is None:
				device_id = torch.cuda.current_device()
		else:
			print('[GPU Memory] Current device is CPU.')
			return

		torch.cuda.synchronize(device_id)

		allocated = torch.cuda.memory_allocated(device_id) / 1024 ** 3
		reserved = torch.cuda.memory_reserved(device_id) / 1024 ** 3
		max_allocated = torch.cuda.max_memory_allocated(device_id) / 1024 ** 3

		free, total = torch.cuda.mem_get_info(device_id)
		free = free / 1024 ** 3
		total = total / 1024 ** 3
		used_all = total - free

		print(
			'[GPU Memory] %s | device: cuda:%d | '
			'PyTorch allocated: %.3f GB | '
			'PyTorch reserved: %.3f GB | '
			'PyTorch max allocated: %.3f GB | '
			'GPU used all: %.3f GB | '
			'GPU free: %.3f GB | '
			'GPU total: %.3f GB'
			% (tag, device_id, allocated, reserved, max_allocated, used_all, free, total)
		)

	def train(self):


		unet_path = os.path.join(self.model_path, '%s-%d-%.4f.pkl' % (self.model_type, self.num_epochs, self.lr))

		if os.path.isfile(unet_path):
			self.unet.load_state_dict(torch.load(unet_path))
			print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
		else:
			learn_rate = self.lr
			best_Dice = 0.

			print("Train......\n")
			for epoch in range(self.num_epochs):

				if torch.cuda.is_available():
					torch.cuda.reset_peak_memory_stats(self.device)

				self.print_gpu_memory('Epoch %d start' % (epoch + 1))

				self.unet.train()
				epoch_loss = 0

				t_TPR = 0.  # TPR
				t_FPR = 0.  # FPR
				t_PPV = 0.  # PPV
				t_JS = 0.   # Jaccard Similarity
				t_DC = 0.   # Dice Coefficient
				t_SE = 0.   # SE
				t_SP = 0.   # SP
				#t_HD = 0.
				# t_Recall = 0.
				mask_array = np.array([
					[True, False],
					[False, True],
					[True, True]
				])


				for i, (images, pet, label) in enumerate(self.train_loader):

					images = images.to(self.device)
					pet = pet.to(self.device)
					GT = label.to(self.device)

					SR = F.sigmoid(self.unet(images, pet))
					SR_flat = SR.view(SR.size(0), -1)
					GT_flat = GT.view(GT.size(0), -1)
					loss = self.criterion(SR_flat, GT_flat)


					epoch_loss += loss.item()

					# Backprop + optimize
					self.reset_grad()
					loss.backward()
					self.optimizer.step()

					if (i + 1) % 10 == 0 or (i + 1) == len(self.train_loader):
						self.print_gpu_memory(
							'Train Epoch [%d/%d], Iter [%d/%d]'
							% (epoch + 1, self.num_epochs, i + 1, len(self.train_loader))
						)

					t_TPR += get_TPR(SR, GT)
					t_FPR += get_FPR(SR, GT)
					t_PPV += get_precision(SR, GT)
					t_JS += get_JS(SR, GT)
					t_DC += get_DC(SR, GT)
					t_SE += get_sensitivity(SR, GT)
					t_SP += get_specificity(SR, GT)


				length = len(self.train_loader)
				t_TPR = t_TPR / length
				t_FPR = t_FPR / length
				t_PPV = t_PPV / length
				t_JS = t_JS / length
				t_DC = t_DC / length
				t_SE = t_SE / length
				t_SP = t_SP / length


				print('Epoch [%d/%d], \n[Training]   TPR: %.4f, FPR: %.4f, PPV:%.4f, JS: %.4f, DC: %.4f, SE:%.4f, SP:%.4f'    # , Recall:%.4f
				      % (epoch + 1, self.num_epochs, t_TPR, t_FPR, t_PPV, t_JS, t_DC, t_SE, t_SP))   # , t_HD, t_Recall

				self.print_gpu_memory('Epoch %d after training' % (epoch + 1))
				e = open(os.path.join(self.train_result_path, 'train_result.csv'), 'a', encoding='utf-8',newline='')
				wr = csv.writer(e)
				wr.writerow([self.model_type, t_TPR, t_FPR, t_PPV, t_JS, t_DC, t_SE, t_SP, epoch_loss, epoch + 1, self.num_epochs, learn_rate])   # , t_HD, t_Recall
				e.close()

	#===================================== Validation ====================================#
				print("Valid......\n")
				self.unet.train(False)
				self.unet.eval()

				with torch.no_grad():
					v_TPR = 0.  # TPR
					v_FPR = 0.  # FPR
					v_PPV = 0.  # PPV
					v_JS = 0.  # Jaccard Similarity
					v_DC = 0.  # Dice Coefficient
					v_SE = 0.  # SE
					v_SP = 0.  # SP


					for i, (images, pet,  GT) in enumerate(self.valid_loader):

						images = images.to(self.device)
						pet = pet.to(self.device)
						GT = GT.to(self.device)

						SR = F.sigmoid(self.unet(images, pet))


						v_TPR += get_TPR(SR, GT)
						v_FPR += get_FPR(SR, GT)
						v_PPV += get_precision(SR, GT)
						v_JS += get_JS(SR, GT)
						v_DC += get_DC(SR, GT)
						v_SE += get_sensitivity(SR, GT)
						v_SP += get_specificity(SR, GT)


					length = len(self.valid_loader)
					v_TPR = v_TPR / length
					v_FPR = v_FPR / length
					v_PPV = v_PPV / length
					v_JS = v_JS / length
					v_DC = v_DC / length
					v_SE = v_SE / length
					v_SP = v_SP / length

					# Print the log info
					print('[Validation] TPR: %.4f, FPR: %.4f, PPV:%.4f, JS: %.4f, DC: %.4f, SE:%.4f, SP:%.4f\n[Loss]: %.4f  [lr]: %.4f'   # , Recall:%.4f
					      % (v_TPR, v_FPR, v_PPV, v_JS, v_DC, v_SE, v_SP, epoch_loss, learn_rate))     # , v_HD, v_Recall

					

					h = open(os.path.join(self.val_result_path, 'val_result.csv'), 'a', encoding='utf-8', newline='')
					wr = csv.writer(h)
					wr.writerow([self.model_type, v_TPR, v_FPR, v_PPV, v_JS, v_DC, v_SE, v_SP, epoch + 1, self.num_epochs])    # , v_HD, v_Recall
					h.close()

					# Save Best U-Net model
					if v_DC + v_JS > best_Dice:
						best_Dice = v_DC + v_JS
						best_unet = self.unet.state_dict()
						print('Best %s model score : %.4f' % (self.model_type, best_Dice))
						torch.save(best_unet, unet_path)


	# ===================================== Test ====================================#
	def test(self):

		unet_path = os.path.join(
			self.model_path,
			'%s-%d-%.4f.pkl' % (self.model_type, self.num_epochs, self.lr)
		)

		self.unet.load_state_dict(torch.load(unet_path))
		print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))

		self.unet.eval()

		fold_infer_time = 0.0
		fold_num_samples = 0

		with torch.no_grad():
			for epoch in range(self.num_epochs_test):

				TPR = 0.
				FPR = 0.
				PPV = 0.
				JS = 0.
				DC = 0.
				SE = 0.
				SP = 0.
				HD = 0.
				Recall = 0.

				for i, (images, pet, GT) in enumerate(self.test_loader):

					images = images.to(self.device)
					pet = pet.to(self.device)
					GT = GT.to(self.device)

					if torch.cuda.is_available():
						torch.cuda.synchronize(self.device)

					infer_start = time.perf_counter()

					SR = torch.sigmoid(self.unet(images, pet))

					if torch.cuda.is_available():
						torch.cuda.synchronize(self.device)

					infer_end = time.perf_counter()

					fold_infer_time += infer_end - infer_start
					fold_num_samples += images.size(0)

					# ================== 指标计算，不计入推理时间 ==================
					TPR += get_TPR(SR, GT)
					FPR += get_FPR(SR, GT)
					PPV += get_precision(SR, GT)
					JS += get_JS(SR, GT)
					DC += get_DC(SR, GT)
					SE += get_sensitivity(SR, GT)
					SP += get_specificity(SR, GT)
					HD += get_HD(SR, GT)
					Recall += get_Recall(SR, GT)

				length = len(self.test_loader)
				TPR = TPR / length
				FPR = FPR / length
				PPV = PPV / length
				JS = JS / length
				DC = DC / length
				SE = SE / length
				SP = SP / length
				HD = HD / length
				Recall = Recall / length

				print(
					'Epoch [%d/%d], \n[Test] TPR: %.4f, FPR: %.4f, PPV: %.4f, JS: %.4f, DC: %.4f, SE: %.4f, SP: %.4f, HD: %.4f, Recall: %.4f'
					% (
						epoch + 1,
						self.num_epochs_test,
						TPR,
						FPR,
						PPV,
						JS,
						DC,
						SE,
						SP,
						HD,
						Recall
					)
				)

				g = open(
					os.path.join(self.test_result_path, 'test_result.csv'),
					'a',
					encoding='utf-8',
					newline=''
				)
				wr = csv.writer(g)
				wr.writerow([
					self.model_type,
					TPR,
					FPR,
					PPV,
					JS,
					DC,
					SE,
					SP,
					HD,
					Recall,
					self.lr,
					epoch + 1,
					self.num_epochs_test
				])
				g.close()

		fold_avg_time_per_image = fold_infer_time / fold_num_samples
		fold_fps = fold_num_samples / fold_infer_time

		return fold_infer_time, fold_avg_time_per_image, fold_fps, fold_num_samples

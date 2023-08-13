from pytorch_fid.fid_score import *
import torch

# list all available metrics

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# # create metric with default setting
# iqa_metric = pyiqa.create_metric('lpips', device=device)
# # Note that gradient propagation is disabled by default. set as_loss=True to enable it as a loss function.
# iqa_loss = pyiqa.create_metric('lpips', device=device, as_loss=True)

# # create metric with custom setting
# iqa_metric = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device)

# # check if lower better or higher better
# print(iqa_metric.lower_better)

# # example for iqa score inference
# # Tensor inputs, img_tensor_x/y: (N, 3, H, W), RGB, 0 ~ 1
# score_fr = iqa_metric(img_tensor_x, img_tensor_y)
# score_nr = iqa_metric(img_tensor_x)

# # img path as inputs.
# score_fr = iqa_metric('./IQA-PyTorch/ResultsCalibra/dist_dir/I03.bmp', './IQA-PyTorch/ResultsCalibra/ref_dir/I03.bmp')

# For FID metric, use directory or precomputed statistics as inputs
# refer to clean-fid for more details: https://github.com/GaParmar/clean-fid
# fid_metric = pyiqa.create_metric('fid')
score = calculate_fid_given_paths(['./evaluate/IQA-PyTorch/ResultsCalibra/dist_dir/', './evaluate/IQA-PyTorch/ResultsCalibra/ref_dir'], batch_size=1, device=device, dims=2048)
print(score)
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.transforms as transforms
from utils.image_transforms import ArrayToTensor

def prepare_transforms():
    # transforms
    source_img_transforms = A.Compose([
        A.ColorJitter(p=0.5),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.5),
        A.OneOf([A.ChannelDropout(p=0.5),
                 A.ChannelShuffle(p=0.5),
                 A.ToGray(p=0.5),
                 A.ToSepia(p=0.5)]),
        A.OneOf([A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.5, p=0.5),
                 A.RandomRain(p=0.5),
                 A.RandomSnow(p=0.5),
                 A.RandomSunFlare(src_radius=150, p=0.5)]),
        A.OneOf([A.Compose([A.CropAndPad(percent=-0.07), A.CropAndPad(percent=0.07)]),
                 A.Compose([A.CropAndPad(percent=-0.03), A.CropAndPad(percent=0.03)]),
                 A.Compose([]),
                 ]),
        ToTensorV2()])
    target_img_transforms = A.Compose([
        A.ColorJitter(p=0.5),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.5),
        A.OneOf([A.ChannelDropout(p=0.5),
                 A.ChannelShuffle(p=0.5),
                 A.ToGray(p=0.5),
                 A.ToSepia(p=0.5)]),
        A.OneOf([A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.7, p=0.5),
                 A.RandomRain(p=0.5),
                 A.RandomSnow(p=0.5),
                 A.RandomSunFlare(src_radius=150, p=0.5)]),
        A.OneOf([A.Compose([A.CropAndPad(percent=-0.07), A.CropAndPad(percent=0.07)]),
                 A.Compose([A.CropAndPad(percent=-0.03), A.CropAndPad(percent=0.03)]),
                 A.Compose([]),
                 ]),
        ToTensorV2()])
    co_transform = None
    # for panoramic images
    # co_transform = A.Compose([
    #     A.RandomCrop(height=224,width=320),
    #     A.Resize(height=480,width=640)
    # ])
    flow_transform = transforms.Compose([ArrayToTensor()])  # just put channels first and put it to float
    change_transform = transforms.Compose([ArrayToTensor()])

    return source_img_transforms, target_img_transforms, co_transform, flow_transform, change_transform
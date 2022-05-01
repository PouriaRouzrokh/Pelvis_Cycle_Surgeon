#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

# Standard built-in modules
import warnings

# Third-party modules
import torch
import torchvision

#-------------------------------------------------------------------------------
# Configurations
#-------------------------------------------------------------------------------

warnings.filterwarnings("ignore") 

#-------------------------------------------------------------------------------
# Custom metric functions
#-------------------------------------------------------------------------------

#-----------------------------------
# - C: VGG_PerceptualLoss

class VGG_PerceptualLoss():
    def __init__(self, style_weight=100, content_weight=1, device='cpu',
                 tv_loss_weight=1e-5, layers=[4, 9, 16, 23], 
                 layer_weights=[1, 1, 1, 1]):
        self.vgg_model = torchvision.models.vgg16(pretrained=True)
        self.vgg_model.to(device)
        self.layers = layers
        self.layer_weights = [torch.tensor(x) for x in layer_weights]
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.tv_loss_weight = tv_loss_weight
        self.mse_loss = torch.nn.MSELoss()
    
    def normalize_to_imagenet(self, img):
        imagenet_mean = (torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        imagenet_mean = imagenet_mean.to(device=img.device)
        imagenet_std = (torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        imagenet_std = imagenet_std.to(device=img.device)
        img -= imagenet_mean
        img /= imagenet_std
        return img
    
    def get_features(self, real_imgs, fake_imgs, layer_number):
        self.vgg_model.eval()
        model_chunk = torch.nn.Sequential(
            *list(self.vgg_model.features)[:layer_number])
        for param in model_chunk.parameters():
            param.requires_grad = False
        real_imgs = self.normalize_to_imagenet(torch.cat([real_imgs]*3, dim=1))
        fake_imgs = self.normalize_to_imagenet(torch.cat([fake_imgs]*3, dim=1))   
        r_features = model_chunk(real_imgs)
        f_features = model_chunk(fake_imgs)
        return r_features, f_features
    
    def get_gram_matrix(self, tensor):
        b, c, _, _ = tensor.size()
        tensor = tensor.view(b * c, -1)
        gram = torch.mm(tensor, tensor.t())
        return gram
    
    def get_style_loss(self, r_features, f_features):
        _, c, h, w = r_features.shape
        r_gram_matrix = self.get_gram_matrix(r_features)
        f_gram_matrix = self.get_gram_matrix(f_features)
        return self.mse_loss(r_gram_matrix, f_gram_matrix) / (c * h * w)
    
    def get_content_loss(self, r_features, f_features):
        return self.mse_loss(r_features, f_features)
    
    def get_tv_loss(self, img):
        w_variance = torch.sum(torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2))
        h_variance = torch.sum(torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2))
        loss = self.tv_loss_weight * (h_variance + w_variance)
        return loss
    
    def __call__(self, real_imgs, fake_imgs):
        total_loss = torch.tensor(0.0, device=real_imgs.device)
        for i, w in zip(self.layers, self.layer_weights):
            r_features, f_features = self.get_features(real_imgs, fake_imgs, i)
            style_loss = self.get_style_loss(r_features, f_features)
            content_loss = self.get_content_loss(r_features, f_features)
            total_loss += self.style_weight * style_loss * w
            total_loss += self.content_weight * content_loss * w
        
        return total_loss + self.get_tv_loss(fake_imgs)
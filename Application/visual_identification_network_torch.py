import torch
import torch.nn as nn
from torchvision import transforms

# Define Normalize Layer
class Normalize(nn.Module):
    def __init__(self, device):
        super(Normalize, self).__init__()
        self.norm = None
        self.device = device
        
    def forward(self, x):
        if self.norm is None:
            channels = x.size(1)
            self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406][:channels], std=[0.229, 0.224, 0.225][:channels])
        #print(f'x: {x.size()} output shape: {self.norm.output_shape}')
        return self.norm(x / 255.0)
        #return (x / 127.5) - 1.0
        #return x

# %%
# Define v200 model
class V200(nn.Module):
    def __init__(self, image_width, image_height, num_classes, channels):
        super(V200, self).__init__()
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3)
        self.dropout1 = nn.Dropout2d(0.05)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding='same')
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3)
        self.dropout2 = nn.Dropout2d(0.25)

        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding='same')
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=3)
        self.dropout3 = nn.Dropout2d(0.05)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(512, 1024)
        self.bn6 = nn.BatchNorm1d(1024)
        self.relu6 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.05)

        self.fc2 = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        #from torch.mps import profiler

        #with profiler.profile() as prof:
        if True:
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.pool1(self.relu2(self.bn2(self.conv2(x))))
            x = self.dropout1(x)

            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.pool2(self.relu4(self.bn4(self.conv4(x))))
            x = self.dropout2(x)

            x = self.relu5(self.bn5(self.conv5(x)))
            x = self.pool3(x)
            x = self.dropout3(x)

            x = self.global_avg_pool(x)
            x = x.view(x.size(0), -1)  # Flatten

            x = self.relu6(self.bn6(self.fc1(x)))
            x = self.dropout4(x)
            x = self.fc2(x)
        
        return x

# %%
# Define v119 model
class V119(nn.Module):
    def __init__(self, image_width, image_height, num_classes, channels):
        super(V119, self).__init__()
        self.conv1 = nn.Conv2d(channels, 256, kernel_size=5, padding='same')
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout2d(0.05)

        self.conv2 = nn.Conv2d(256, 128, kernel_size=5, padding='same')
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout2d(0.05)

        self.conv3 = nn.Conv2d(128, 32, kernel_size=5, padding='same')
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.dropout3 = nn.Dropout2d(0.05)

        self.conv4 = nn.Conv2d(32, 128, kernel_size=5, padding='same')
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.dropout4 = nn.Dropout2d(0.05)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * (image_width // 16) * (image_height // 16), 1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Permute to channels first (N, H, W, C) -> (N, C, H, W)
        #x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        x = self.dropout4(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.fc2(x)
        return x

# %%
# Define v118_3 model
class V118_3(nn.Module):
    def __init__(self, image_width, image_height, num_classes, channels):
        super(V118_3, self).__init__()
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=5, padding='same')
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout2d(0.05)

        self.conv2 = nn.Conv2d(16, 64, kernel_size=5, padding='same')
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout2d(0.05)

        self.conv3 = nn.Conv2d(64, 100, kernel_size=5, padding='same')
        self.bn3 = nn.BatchNorm2d(100)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.dropout3 = nn.Dropout2d(0.05)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(100 * (image_width // 8) * (image_height // 8), 100)
        self.bn4 = nn.LayerNorm(100)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.05)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        #x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        #x = self.relu3(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn4(x)
        x = self.relu4(x)
        #x = self.relu4(self.bn4(self.fc1(x)))
        x = self.dropout4(x)
        x = self.fc2(x)
        return x

# %%
# Define v110 model
class V110(nn.Module):
    def __init__(self, image_width, image_height, num_classes, channels):
        super(V110, self).__init__()
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=5, padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(0.25)

        self.conv2 = nn.Conv2d(16, 64, kernel_size=5, padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(0.25)

        self.conv3 = nn.Conv2d(64, 100, kernel_size=5, padding='same')
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.bn3 = nn.BatchNorm2d(100)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout2d(0.25)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(100 * (image_width // 8) * (image_height // 8), 100)
        self.bn4 = nn.BatchNorm1d(100)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.dropout4(x)
        x = self.fc2(x)
        return x

# %%
# Define v100 model
class V100(nn.Module):
    def __init__(self, image_width, image_height, num_classes, channels):
        super(V100, self).__init__()
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=5, padding='same')
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout2d(0.25)

        self.conv2 = nn.Conv2d(16, 64, kernel_size=5, padding='same')
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout2d(0.25)

        self.conv3 = nn.Conv2d(64, 100, kernel_size=5, padding='same')
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.dropout3 = nn.Dropout2d(0.25)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(100 * (image_width // 8) * (image_height // 8), 100)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout4(x)
        x = self.fc2(x)
        return x

# %%
import torch
import torch.nn as nn
import torchvision.models as models

class ModelFetcher:
    def __init__(self):
        self.versions = {}
        self.add_official_models()
        self.add_custom_models()

    def add_official_models(self):
        def top_for(model, num_classes, resize=None):
            # Freeze the pretrained weights
            for param in model.parameters():
                param.requires_grad = True

            # Modify the final fully connected layer to match num_classes
            if hasattr(model, 'fc'):
                print(f"model.fc: {model.fc}")
                num_features = model.fc.in_features
                model.fc = nn.Linear(num_features, num_classes)
                #model.fc = nn.Sequential(
                #    nn.Flatten(),
                #    nn.Linear(num_features, num_classes),
                #    nn.Softmax(dim=1)
                #)
            elif hasattr(model, 'classifier'):
                print(model.classifier)
                for i in range(len(model.classifier)):
                    j = len(model.classifier) - i - 1

                    if isinstance(model.classifier[j], nn.Linear):
                        num_features = model.classifier[j].in_features
                        print(f"num_features: {num_features}")
                        print(f"classifier: {model.classifier[j]}")
                        model.classifier[j] = nn.Linear(num_features, num_classes)
                        break

            if resize is not None:
                model = nn.Sequential(
                    nn.Upsample(size=resize, mode='bilinear'),
                    model
                )
            return model

        def modify_first_layer(model, channels):
            if channels != 3:
                if hasattr(model, 'features'):
                    if hasattr(model.features, '0'):
                        print(f"model.features[0]: {model.features[0]}")
                        if type(model.features[0]) == nn.Conv2d:
                            if hasattr(model, 'avgpool'):
                                print(f"model.avgpool: {model.avgpool}")
                                print(f"model.features[-1]: {model.features[-3]}")
                                print(f"model.classifier[0]: {model.classifier[0]}")
                                #model.avgpool = nn.Sequential(
                                #    nn.AdaptiveAvgPool2d((1, 1)),
                                #    nn.Linear(model.features[-3].out_channels, model.classifier[0].in_features),
                                #)
                                #model.avgpool = nn.AdaptiveAvgPool2d((196, 32))

                            model.features[0] = nn.Conv2d(channels, model.features[0].out_channels, kernel_size=model.features[0].kernel_size, stride=model.features[0].stride, padding=model.features[0].padding, bias=False)
                        else:
                            model.features[0][0] = nn.Conv2d(channels, model.features[0][0].out_channels, kernel_size=model.features[0][0].kernel_size, stride=model.features[0][0].stride, padding=model.features[0][0].padding, bias=False)
                elif hasattr(model, 'conv1'):
                    model.conv1 = nn.Conv2d(channels, model.conv1.out_channels, kernel_size=model.conv1.kernel_size, stride=model.conv1.stride, padding=model.conv1.padding, bias=False)
                elif hasattr(model, 'Conv2d_1a_3x3'):
                    print(f"model.Conv2d_1a_3x3: {model.Conv2d_1a_3x3}")
                    model.Conv2d_1a_3x3.conv = nn.Conv2d(
                        channels, 
                        model.Conv2d_1a_3x3.conv.out_channels, 
                        kernel_size=model.Conv2d_1a_3x3.conv.kernel_size, 
                        stride=model.Conv2d_1a_3x3.conv.stride, 
                        padding=model.Conv2d_1a_3x3.conv.padding, 
                        bias=False)
            return model

        def convnext_base(num_classes, channels):
            model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
            model = modify_first_layer(model, channels)
            return top_for(model, num_classes)

        self.versions["convnext_base"] = convnext_base

        def inception_v3(num_classes, channels):
            model = models.inception_v3(#weights=models.Inception_V3_Weights.DEFAULT, 
                                        num_classes=num_classes, aux_logits=False)
            model = modify_first_layer(model, channels)
            return top_for(model, num_classes, resize=(299,299))

        self.versions["inception_v3"] = inception_v3

        def vgg_16(num_classes, channels):
            model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            model = modify_first_layer(model, channels)
            return top_for(model, num_classes)

        self.versions["vgg_16"] = vgg_16

        def vgg_19(num_classes, channels):
            model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
            model = modify_first_layer(model, channels)
            return top_for(model, num_classes)

        self.versions["vgg_19"] = vgg_19

        def mobilenet_v3_small(num_classes, channels):
            model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            model = modify_first_layer(model, channels)
            return top_for(model, num_classes)

        self.versions["mobilenet_v3_small"] = mobilenet_v3_small

        def mobilenet_v3_large(num_classes, channels):
            model = models.mobilenet_v3_large(
                #weights=models.MobileNet_V3_Large_Weights.DEFAULT
            )
            model = modify_first_layer(model, channels)
            return top_for(model, num_classes)

        self.versions["mobilenet_v3_large"] = mobilenet_v3_large

        def resnet_50_v2(num_classes, channels):
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            model = modify_first_layer(model, channels)
            return top_for(model, num_classes)

        self.versions["resnet_50_v2"] = resnet_50_v2

        def efficientnet_b0(num_classes, channels):
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            model = modify_first_layer(model, channels)
            return top_for(model, num_classes)

        self.versions["efficientnet_b0"] = efficientnet_b0

        def resnet_18(num_classes, channels):
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            model = modify_first_layer(model, channels)
            return top_for(model, num_classes)
        
        self.versions["resnet_18"] = resnet_18

        self.array_of_all_official_models = [
            "convnext_base", "vgg_16", "vgg_19", "mobilenet_v3_small", "mobilenet_v3_large", 
            "resnet_50_v2", "efficientnet_b0", "inception_v3", "resnet_18"
        ]

    def add_custom_models(self):
        def v200(num_classes, channels, image_width, image_height):
            return V200(image_width=image_width, image_height=image_height, num_classes=num_classes, channels=channels)
        
        self.versions["v200"] = v200

        def v119(num_classes, channels, image_width, image_height):
            return V119(image_width=image_width, image_height=image_height, num_classes=num_classes, channels=channels)

        self.versions["v119"] = v119

        def v118_3(num_classes, channels, image_width, image_height):
            return V118_3(image_width=image_width, image_height=image_height, num_classes=num_classes, channels=channels)

        self.versions["v118_3"] = v118_3

        def v100(num_classes, channels, image_width, image_height):
            return V100(image_width=image_width, image_height=image_height, num_classes=num_classes, channels=channels)

        self.versions["v100"] = v100

        def v110(num_classes, channels, image_width, image_height):
            return V110(image_width=image_width, image_height=image_height, num_classes=num_classes, channels=channels)

        self.versions["v110"] = v110

        self.array_of_custom_models = [
            "v119", "v118_3", "v100", "v110"
        ]

    def get_model(self, model_name, num_classes, channels, image_width=None, image_height=None, device=None):
        if model_name in self.versions:
            if model_name in self.array_of_all_official_models:
                model = self.versions[model_name](num_classes, channels)
            else:
                model = self.versions[model_name](num_classes, channels, image_width, image_height)
            model.to(device)
            return PermuteAxesWrapper(model, device=device)
        else:
            raise ValueError(f"Model {model_name} not found. Available models are: {list(self.versions.keys())}")

    @staticmethod
    def save_model(model, filepath):
        torch.save(model.state_dict(), filepath)

    @staticmethod
    def load_model(model, filepath):
        model.load_state_dict(torch.load(filepath, weights_only=True))
        model.eval()
        return model

class PermuteAxesWrapper(nn.Module):
    def __init__(self, model, device):
        super(PermuteAxesWrapper, self).__init__()
        self.model = model
        self.normalize = Normalize(device=device)

    def forward(self, x):
        # Permute to channels first (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2)                        
        x = self.normalize(x)
        return self.model(x)

'''
    # Example usage
    model_fetcher = ModelFetcher()
    model_name = "resnet_18"
    num_classes = 10
    channels = 3  # For grayscale images
    image_width = 128
    image_height = 128
    device = 'mps'

    model = model_fetcher.get_model(model_name, num_classes, channels, image_width, image_height, device=device)
    print(model)

    # Example input
    input_tensor = torch.randn(16, image_height, image_width, channels).to(device)  # Batch size of 16, image size 224x224, 1 channel
    output = model(input_tensor)
    #print(output)
    print(output.shape, "trainable parameters = ", count_parameters(model))  # Should be (16, num_classes)
'''
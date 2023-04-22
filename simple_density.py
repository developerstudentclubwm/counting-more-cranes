import torch
import torch.nn as nn
import torchvision.models as models

class MultiTaskAU(nn.Module):
    #color channel, height, width 
    def __init__(self, input_shape=(3, 200, 200)):
        super(MultiTaskAU, self).__init__()
        #initializes ResNet model 
        self.resnet = models.resnet50(pretrained=True)
        #converts to sequential model
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        #conv layer with 2048 input channels, 2 output channels, and kernel size of 1
        #performs conv on ResNet output 
        self.conv = nn.Conv2d(2048, 2, kernel_size=1)
        #avg pooling layer
        #ouputs tensor of batch size and number of channels
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.sub = torch.sub

    def forward(self, input_count, input_pair1, input_pair2):
        output_count = self.resnet(input_count)
        output_pair1 = self.resnet(input_pair1)
        output_pair2 = self.resnet(input_pair2)

        output_count = self.conv(output_count)
        output_pair1 = self.conv(output_pair1)
        output_pair2 = self.conv(output_pair2)

        output_pair1 = self.global_pool(output_pair1)
        output_pair2 = self.global_pool(output_pair2)
        pair_diff = self.sub(output_pair2, output_pair1)

        return output_count, pair_diff
    
#testing 
if __name__ == '__main__':
    #just for testing 
    #add to full_pipeline.py if functions correctly 




    

    model = MultiTaskAU()
    #test with random tensors 
    input_count = torch.randn(1, 3, 200, 200)
    input_pair1 = torch.randn(1, 3, 200, 200)
    input_pair2 = torch.randn(1, 3, 200, 200)

    output_count, pair_diff = model(input_count, input_pair1, input_pair2)
    #output_count = model(input_count, input_pair1, input_pair2)
    print(output_count)

    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from PIL import Image

    mosaic_path = '/home/cropthecoder/Documents/Counting-Crane/counting-more-cranes/20180320_212958_600_9152.tif'
    mosaic = Image.open(mosaic_path)

    #ask about image resizing 

    # Convert the image to a tensor and normalize the pixel values
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mosaic_tensor = transform(mosaic)

    # Add an extra batch dimension to the tensor
    mosaic_tensor = mosaic_tensor.unsqueeze(0)

    print(mosaic_tensor)
    




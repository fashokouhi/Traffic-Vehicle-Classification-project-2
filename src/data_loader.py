from torch.utils.data import  Dataset,DataLoader
from torchvision import  transforms
from PIL import Image


class VehicleDataset(Dataset):
    def __init__(self, image_paths,labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform


    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self,index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image)
        return image , label



def create_dataloaders(
    X_train,
    X_val,
    y_train,
    y_val,
    batch_size = 32
):


    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])


    train_dataset = VehicleDataset(
        image_paths=X_train,
        labels=y_train,
        transform=transform
    )


    val_dataset = VehicleDataset(
        image_paths=X_val,
        labels=y_val,
        transform=transform
    )

    train_loader = DataLoader(
    train_dataset,
    batch_size= batch_size,
    shuffle=True,
    num_workers=0
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader , val_loader
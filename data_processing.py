import random
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
from torch.utils.data import TensorDataset


# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 数据增强
augmentation_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 普通数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 自定义数据集
class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root=root_dir, transform=transform)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

# 数据增强函数
def augment_sample(image, n_augmentations, transform):
    augmented_images = []
    for _ in range(n_augmentations):
        image_pil = Image.fromarray(np.array(image)).convert('RGB')
        augmented_image = transform(image_pil)
        augmented_images.append(augmented_image)
    return augmented_images


# 自定义 TensorDataset，增加 get_labels 方法
class CustomTensorDataset(TensorDataset):
    def __init__(self, *tensors, labels):
        super().__init__(*tensors)
        self.labels = labels

    def get_labels(self):
        return self.labels.tolist()

    def __getitem__(self, index):
        # 返回图像和标签
        return self.tensors[0][index], self.labels[index]



def create_datasets(class_to_indices, my_dataset, total_augmented_images_per_class):
    real_train_images = []
    real_train_labels = []
    augmented_images = []
    augmented_labels = []
    test_images = []
    test_labels = []

    for label, indices in class_to_indices.items():
        selected_indices = random.sample(indices, min(5, len(indices)))
        
        for selected_index in selected_indices:
            image, _ = my_dataset[selected_index]
            real_train_images.append(image)
            real_train_labels.append(label)

        for selected_index in selected_indices:
            image, _ = my_dataset[selected_index]
            image_pil = Image.open(my_dataset.dataset.imgs[selected_index][0])
            augmented_samples = augment_sample(image_pil, n_augmentations=total_augmented_images_per_class - 1, transform=augmentation_transform)
            augmented_images.append(image)
            augmented_images.extend(augmented_samples)
            augmented_labels.extend([label] * total_augmented_images_per_class)

        remaining_indices = [i for i in indices if i not in selected_indices]
        for index in remaining_indices:
            image, _ = my_dataset[index]
            test_images.append(image)
            test_labels.append(label)

    return (
        CustomTensorDataset(torch.stack(augmented_images), labels=torch.tensor(augmented_labels)),
        CustomTensorDataset(torch.stack(real_train_images), labels=torch.tensor(real_train_labels)),
        CustomTensorDataset(torch.stack(test_images), labels=torch.tensor(test_labels))
    )


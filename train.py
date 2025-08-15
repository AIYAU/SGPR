import torch
from tqdm import tqdm

def training_epoch(clip_model, fine_network, train_loader, optimizer, text_prototypes, contrastive_loss_weight=0.1, margin=1.0):
    """
    单个训练 epoch 方法。
    Args:
        clip_model: 冻结的 CLIP 模型，用于特征提取。
        fine_network: 微调网络，用于调整特征。
        train_loader: 训练数据加载器。
        optimizer: 优化器，只更新 Fine Network 的参数。
        text_prototypes: 文本原型 (tensor) [num_classes, feature_dim]。
        contrastive_loss_weight: 对比损失的权重。
        margin: 对比损失的最小分离距离。
    Returns:
        本次 epoch 的平均训练损失。
    """
    fine_network.train()
    criterion = torch.nn.CrossEntropyLoss()  # 分类任务的损失函数
    all_losses = []

    for support_images, support_labels, query_images, query_labels, _ in tqdm(train_loader, desc="Training"):
        # 数据移动到 GPU
        support_images = support_images.to("cuda")
        query_images = query_images.to("cuda")
        support_labels = support_labels.to("cuda")
        query_labels = query_labels.to("cuda")

        # 冻结 CLIP 模型，只提取初始特征
        with torch.no_grad():
            support_features = clip_model.visual(support_images).float()  # 支持集初始特征
            query_features = clip_model.visual(query_images).float()  # 查询集初始特征

        # 使用 Fine Network 提取微调特征
        fine_support_features = fine_network(support_features)
        fine_query_features = fine_network(query_features)

        # 特征结合：微调特征 + 初始特征
        w = 2.0
        combined_support_features =  support_features + w * fine_support_features
        combined_query_features =  query_features + w * fine_query_features

        # 计算类别原型
        prototypes = []
        for class_idx in range(len(set(support_labels.tolist()))):
            class_mask = (support_labels == class_idx)
            if class_mask.sum() > 0:
                prototypes.append(combined_support_features[class_mask].mean(dim=0))  # 当前类别的原型
            else:
                prototypes.append(torch.zeros_like(combined_support_features[0]))
        prototypes = torch.stack(prototypes)

        # 计算对比损失
        contrastive_loss_value = contrastive_loss(prototypes, text_prototypes, margin=margin)

        # 计算查询样本与类别原型的欧式距离
        distances = torch.cdist(combined_query_features, prototypes)
        negative_distances = -distances

        # 计算分类损失
        classification_loss = criterion(negative_distances, query_labels)

        # 总损失：分类损失 + 对比损失
        total_loss = classification_loss + contrastive_loss_weight * contrastive_loss_value

        # 优化 Fine Network
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        all_losses.append(total_loss.item())

    return sum(all_losses) / len(all_losses)


def contrastive_loss(image_prototypes, text_prototypes, margin=1.0):
    """
    对比损失函数，用于对齐图像原型和文本原型，同时分离不同类别。
    Args:
        image_prototypes: 图像原型 (tensor) [num_classes, feature_dim]
        text_prototypes: 文本原型 (tensor) [num_classes, feature_dim]
        margin: 用于分离不同类别的最小距离。
    Returns:
        对比损失值 (tensor)。
    """
    num_classes = image_prototypes.size(0)
    loss = 0.0

    for i in range(num_classes):
        for j in range(num_classes):
            # 计算两个原型的欧式距离
            distance = torch.norm(image_prototypes[i] - text_prototypes[j], p=2)

            if i == j:  # 同类别
                loss += distance ** 2
            else:  # 不同类别
                loss += torch.max(torch.tensor(0.0, device=distance.device), margin - distance) ** 2

    loss /= (num_classes ** 2)  # 平均损失
    return loss

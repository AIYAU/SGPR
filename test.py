import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import numpy as np


def evaluate_model_with_metrics_and_tsne(
    clip_model,
    fine_network,
    test_loader,
    prototypes,
    epoch,
    device="cuda",
    save_path="./results",
    log_file=None
):
    """
    Evaluate model on the test set, print metrics, save logs, and plot TSNE.

    Args:
        clip_model: The frozen CLIP model used for feature extraction.
        fine_network: Fine-tuning network applied on CLIP features.
        test_loader: DataLoader for the test set.
        prototypes: Image prototypes computed from the training set.
        epoch: Current epoch number.
        device: Device to run evaluation on (default: "cuda").
        save_path: Path to save TSNE plots and logs.
        log_file: File to save evaluation logs.

    Returns:
        Overall accuracy of the model on the test set.
    """
    fine_network.eval()
    all_predictions = []
    all_labels = []
    all_features = []

    os.makedirs(save_path, exist_ok=True)

    # Evaluate on test set
    with torch.no_grad():
        for query_images, query_labels in tqdm(test_loader, desc="Evaluating"):
            query_images, query_labels = query_images.to(device), query_labels.to(device)
            
            # Use CLIP for feature extraction
            query_features = clip_model.visual(query_images).float()
            
            # Apply fine-tuning network
            fine_query_features = fine_network(query_features)
            
            # Combine original and fine-tuned features
            query_features = fine_query_features + query_features

            # Compute distances and predictions
            distances = torch.cdist(query_features, prototypes)
            predictions = distances.argmin(dim=1)

            # Collect metrics
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(query_labels.cpu().numpy())
            all_features.extend(query_features.cpu().numpy())

    # Calculate overall metrics
    overall_accuracy = accuracy_score(all_labels, all_predictions) * 100
    precision = precision_score(all_labels, all_predictions, average='macro') * 100
    recall = recall_score(all_labels, all_predictions, average='macro') * 100
    f1 = f1_score(all_labels, all_predictions, average='macro') * 100

    # Calculate per-class accuracy
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    num_classes = len(prototypes)
    class_accuracies = {}
    for class_idx in range(num_classes):
        class_mask = (all_labels == class_idx)
        if class_mask.sum() > 0:
            class_accuracy = (all_predictions[class_mask] == all_labels[class_mask]).sum() / class_mask.sum() * 100
        else:
            class_accuracy = 0.0  # No samples for this class
        class_accuracies[class_idx] = class_accuracy

    # Save metrics to log file
    if log_file:
        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch + 1}/50\n")
            # f.write(f"Average Loss: Not Available\n")  # If you want to add loss, calculate it externally
            for class_idx, accuracy in class_accuracies.items():
                f.write(f"Class {class_idx + 1}: accuracy {accuracy:.2f}%\n")
            f.write("\n")
            f.write(f"overall_accuracy: {overall_accuracy:.2f}%\n")
            f.write(f"precision: {precision:.2f}%\n")
            f.write(f"recall: {recall:.2f}%\n")
            f.write(f"F1-Score: {f1:.2f}%\n")
            f.write("=" * 50 + "\n")

    # Print metrics
    print(f"Epoch {epoch + 1}/50")
    # print(f"Average Loss: Not Available")  # If you want to add loss, calculate it externally
    for class_idx, accuracy in class_accuracies.items():
        print(f"Class {class_idx + 1}: accuracy {accuracy:.2f}%")
    print(f"overall_accuracy: {overall_accuracy:.2f}%")
    print(f"precision: {precision:.2f}%")
    print(f"recall: {recall:.2f}%")
    print(f"F1-Score: {f1:.2f}%")
    

    # Plot TSNE
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(np.array(all_features))
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=all_labels, cmap='tab10', s=40)
    cbar = plt.colorbar(scatter, ticks=range(len(prototypes)))
    cbar.set_label('Classes')
    cbar.set_ticks(range(len(prototypes)))
    cbar.set_ticklabels(range(1, len(prototypes) + 1))
    # plt.title(f"TSNE Visualization - Epoch {epoch + 1}")
    tsne_path = os.path.join(save_path, f"tsne_epoch_{epoch + 1}_{overall_accuracy}.png")
    plt.savefig(tsne_path, dpi=600)
    plt.close()
    # print(f"TSNE plot saved to {tsne_path}")

    return overall_accuracy

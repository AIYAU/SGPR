import torch
import clip
from torch.utils.data import DataLoader
from data_processing import set_seed, create_datasets, MyDataset, transform
from models import FineNetwork
from train import training_epoch
from test import evaluate_model_with_metrics_and_tsne
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from easyfsl.samplers import TaskSampler
import os



class_names = ['Alternaria leaf spot', 'Blossom blight leaves', 'Brown spot', 'Grey spot', 'Health', 'Mosaic', 'Powdery mildew', 'Rust', 'Scab']


# class_names = [
#     "This is a picture of an Alternaria leaf spot: Irregular brown/yellow spots, darker edges, black mold on underside, irregular lesion lesions.",
#     "This is a picture of a Blossom blight leaves: Dark brown to dried yellow leaves, curled edges, wilting appearance, tissue degradation.",
#     "This is a picture of a Brown spot leaves: Abundant yellowing leaves, brown/black lesions, noticeable edge damage, variable lesion size, leaf health compromised.",
#     "This is a picture of a Grey spot leaves: Green leaves with irregular gray-brown spots, clear margins, speckled, mildew presence indicated.",
#     "This is a picture of Healthy apple leaves: Vibrant green color, smooth texture, no disease spots, glossy, signs of optimal health.",
#     "This is a picture of a Mosaic apple leaves: Mottled yellow/green pattern, uneven color, no distinct spots, leaf pattern disrupted.",
#     "This is a picture of Powdery mildew leaves: White/gray powder, uneven color, full leaf coating, fuzzy.",
#     "This is a picture of Rust apple leaves: Scattered orange-yellow/black spots, varied size/shape, slightly elevated.",
#     "This is a picture of Scab apple leaves: Gray spots, indistinct edges, dark gray veins, variable size/shape, scarring."
# ]


def main():
    set_seed(42)

    # Data loading
    root_dir = './dataset'
    my_dataset = MyDataset(root_dir, transform=transform)
    class_to_indices = {label: [] for _, label in my_dataset.dataset.imgs}
    for idx, (_, label) in enumerate(my_dataset.dataset.imgs):
        class_to_indices[label].append(idx)

    # Dataset split
    augmented_train_dataset, real_train_dataset, test_dataset = create_datasets(
        class_to_indices, my_dataset, total_augmented_images_per_class=100
    )

    # Few-shot task setup
    N_WAY = 9
    N_SHOT = 3
    N_QUERY = 19
    N_TRAIN_TASKS = 10
    train_sampler = TaskSampler(
        dataset=augmented_train_dataset,
        n_way=N_WAY,
        n_shot=N_SHOT,
        n_query=N_QUERY,
        n_tasks=N_TRAIN_TASKS
    )
    train_loader = DataLoader(
        augmented_train_dataset,
        batch_sampler=train_sampler,
        pin_memory=True,
        collate_fn=train_sampler.episodic_collate_fn
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Model initialization
    clip_model, _ = clip.load("ViT-L/14", device="cuda")
    clip_model = clip_model.float()
    for param in clip_model.parameters():
        param.requires_grad = False

    fine_network = FineNetwork(feature_dim=clip_model.visual.output_dim).to("cuda")

    # Precompute text prototypes
    text_inputs = clip.tokenize(class_names).to("cuda")
    with torch.no_grad():
        text_prototypes = clip_model.encode_text(text_inputs).float()

    optimizer = Adam(fine_network.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[30, 50], gamma=0.1)

    log_file = "./results_yes_PA_no_TA/training_log.txt"
    os.makedirs("./results_yes_PA_no_TA", exist_ok=True)
    with open(log_file, "w") as f:
        f.write("Training and Evaluation Log\n")
        f.write("=" * 50 + "\n")

    for epoch in range(50):
        print(f"Epoch {epoch + 1}")

        # Training phase
        training_loss = training_epoch(
            clip_model=clip_model,
            fine_network=fine_network,
            train_loader=train_loader,
            optimizer=optimizer,
            text_prototypes=text_prototypes,
            contrastive_loss_weight=0.1,
            margin=1.0
        )
        scheduler.step()

        # Compute prototypes for support set
        real_train_loader = DataLoader(real_train_dataset, batch_size=len(real_train_dataset), shuffle=False)
        support_features, support_labels = next(iter(real_train_loader))
        support_features = support_features.to("cuda")
        support_labels = support_labels.to("cuda")

        w = 2.0

        with torch.no_grad():
            clip_support_features = clip_model.visual(support_features).float()
            fine_support_features = fine_network(clip_support_features)
            combined_support_features =  clip_support_features + w * fine_support_features

        prototypes = []
        missing_classes = []
        for class_idx in range(len(class_names)):
            class_mask = (support_labels == class_idx)
            if class_mask.sum() > 0:  # If samples for the class exist
                prototypes.append(combined_support_features[class_mask].mean(dim=0))
            else:  # If no samples exist for the class
                missing_classes.append(class_idx)
                prototypes.append(torch.zeros_like(combined_support_features[0]))

        if missing_classes:
            print("Warning: No samples found for the following classes:")
            for idx in missing_classes:
                print(f"  - Class {idx}: {class_names[idx]}")

        prototypes = torch.stack(prototypes)  # [num_classes, feature_dim]  

        # Evaluation phase
        test_accuracy = evaluate_model_with_metrics_and_tsne(
            clip_model=clip_model,
            fine_network=fine_network,
            test_loader=test_loader,
            prototypes=prototypes,
            epoch=epoch,
            device="cuda",
            save_path="./results_yes_PA_no_TA",
            log_file=log_file
        )
        
        print(f"Training Loss: {training_loss:.4f}")
        print("=" * 80)

if __name__ == "__main__":
    main()

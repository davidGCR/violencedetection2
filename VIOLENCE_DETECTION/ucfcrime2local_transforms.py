import torchvision.transforms as transforms

def createTransforms(input_size):
    # Data augmentation and normalization for training
    # Just normalization for validation

    # Train Dataset:  [0.48765567 0.4882961  0.48771954] [0.1199861  0.12041671 0.12118535] [0.11998611 0.12041672 0.12118535]
    # Validation Dataset: [0.50160503 0.5023222 0.50156146][0.13373235 0.13337582 0.13282429][0.13373236 0.13337584 0.13282429]

    data_transforms = {
        "train": transforms.Compose(
            [
                # transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                # transforms.RandomResizedCrop(input_size),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4973763, 0.4973337, 0.4961658], std=[0.14330065, 0.1428894,  0.14266236]) #All Train split
                # transforms.Normalize(mean=[0.48765567, 0.4882961,  0.48771954], std=[0.1199861,  0.12041671, 0.12118535]) #onli train split
                # transforms.Normalize([0.49237782, 0.49160805, 0.48998737], [0.11053326, 0.11088469, 0.11275752] )
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                # transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4973763, 0.4973337, 0.4961658], std=[0.14330065, 0.1428894,  0.14266236]) #All Train split
                # transforms.Normalize(mean=[0.50160503, 0.5023222, 0.50156146], std=[0.13373235, 0.13337582, 0.13282429]) #onli validation split
                # transforms.Normalize([0.49237782, 0.49160805, 0.48998737], [0.11053326, 0.11088469, 0.11275752] )
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(input_size),
                # transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4973763, 0.4973337, 0.4961658], std=[0.14330065, 0.1428894,  0.14266236]) #All Train split
                # transforms.Normalize([0.49237782, 0.49160805, 0.48998737], [0.11053326, 0.11088469, 0.11275752] )
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    }
    return data_transforms